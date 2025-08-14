import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm
import random
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import copy

# Set seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds()
class Config:
    results_dir = 'results_effnet_weighted_ce_1'
    data_root = '/home/pankaj/scratch/GBCU_DS/Comined_DS_1_2/data_cropped_new'
    batch_size = 8
    num_workers = 4
    max_bag_size = 5
    min_bag_size_for_aug = 2
    backbone = 'efficientnet_b4'
    hidden_dim = 512
    num_classes = 2
    dropout = 0.3
    attention_dim = 256
    learning_rate = 2e-4
    weight_decay = 1e-5
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_folds = 5
    patience = 6
    class_weights = [2.0, 1.0]  # Benign:Malignant, increase benign weight to reduce bias
    save_attention_maps = False
    attention_map_dir = 'attention_maps'
    use_tta = False
    tta_flips = 2
    tta_rotations = 2
    gradcam_dir = 'gradcam_val'
    gradcam_n = 1

config = Config()

# Create results directory and subdirectories
if not os.path.exists(config.results_dir):
    os.makedirs(config.results_dir)

models_dir = os.path.join(config.results_dir, 'models')
plots_dir = os.path.join(config.results_dir, 'plots')
attention_maps_dir = os.path.join(config.results_dir, config.attention_map_dir)
gradcam_dir = os.path.join(config.results_dir, config.gradcam_dir)

for directory in [models_dir, plots_dir, attention_maps_dir, gradcam_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def mil_collate_fn(batch):
    images_list = []
    labels = []
    patient_ids = []
    bag_sizes = []
    for item in batch:
        images_list.append(item['images'])
        labels.append(item['label'])
        patient_ids.append(item['patient_id'])
        bag_sizes.append(item['bag_size'])
    return {
        'images': images_list,
        'label': torch.stack(labels),
        'patient_id': patient_ids,
        'bag_size': bag_sizes
    }

class MILDataset(Dataset):
    def __init__(self, patient_list, transform=None, training=True, max_bag_size=20, min_bag_size_for_aug=5):
        self.patient_list = patient_list
        self.transform = transform
        self.training = training
        self.max_bag_size = max_bag_size
        self.min_bag_size_for_aug = min_bag_size_for_aug
        self.labels = [p['label'] for p in patient_list]
        class_counts = Counter(self.labels)
        print(f"Dataset - Benign: {class_counts[0]}, Malignant: {class_counts[1]}")

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient_info = self.patient_list[idx]
        patient_id = patient_info['id']
        label = patient_info['label']
        patient_path = patient_info['path']
        image_files = [f for f in os.listdir(patient_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
        images = []
        for img_file in image_files:
            try:
                img_path = os.path.join(patient_path, img_file)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                continue
        if self.training and len(images) < self.min_bag_size_for_aug and len(images) > 0:
            needed = self.min_bag_size_for_aug - len(images)
            for _ in range(needed):
                src_img = random.choice(images)
                aug = transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.RandomRotation(90),
                ])
                images.append(aug(src_img))
        if len(images) > self.max_bag_size:
            if self.training:
                idxs = random.sample(range(len(images)), self.max_bag_size)
                images = [images[i] for i in idxs]
            else:
                step = max(1, len(images) // self.max_bag_size)
                images = images[::step][:self.max_bag_size]
        if not images:
            images = [torch.ones((3, 380, 380)) * 0.5]
        images = torch.stack(images)
        return {
            'images': images,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'bag_size': len(images)
        }

class GatedAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)
    def forward(self, features):
        V = self.attention_V(features)
        U = self.attention_U(features)
        attention_scores = self.attention_weights(V * U)
        attention = F.softmax(attention_scores, dim=0)
        return attention

class GatedAttentionMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.feature_dim = self.backbone.num_features
        self.feature_net = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim)
        )
        self.attention = GatedAttention(
            input_dim=config.hidden_dim,
            attention_dim=config.attention_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward_bag(self, images):
        device = next(self.parameters()).device
        images = images.to(device)
        features = self.backbone(images)
        transformed_features = self.feature_net(features)
        attention_weights = self.attention(transformed_features)
        weighted_features = torch.sum(transformed_features * attention_weights, dim=0, keepdim=True)
        logits = self.classifier(weighted_features)
        return logits, attention_weights, transformed_features
    def forward(self, images_list):
        batch_logits = []
        batch_attentions = []
        batch_features = []
        for images in images_list:
            logits, attention, features = self.forward_bag(images)
            batch_logits.append(logits)
            batch_attentions.append(attention)
            batch_features.append(features)
        batch_logits = torch.cat(batch_logits, dim=0)
        return batch_logits, batch_attentions, batch_features

def prepare_patient_list(data_root):
    patient_list = []
    for class_name in ['benign', 'malignant']:
        class_path = os.path.join(data_root, class_name)
        if not os.path.exists(class_path):
            continue
        label = 0 if class_name == 'benign' else 1
        for patient_folder in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_folder)
            if os.path.isdir(patient_path):
                image_files = [f for f in os.listdir(patient_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
                if len(image_files) > 0:
                    patient_list.append({
                        'id': f"{class_name}/{patient_folder}",
                        'path': patient_path,
                        'label': label,
                        'num_images': len(image_files)
                    })
    return patient_list

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    class_preds_count = {0: 0, 1: 0}
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Training')):
        images_list = batch['images']
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits, _, _ = model(images_list)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        for p in preds.cpu().numpy():
            class_preds_count[p] = class_preds_count.get(p, 0) + 1
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Train Loss: {running_loss / len(dataloader.dataset):.4f} | "
          f"Accuracy: {accuracy:.4f} | Balanced Acc: {balanced_acc:.4f} | F1: {f1:.4f}")
    print(f"Training predictions - Benign: {class_preds_count[0]}, Malignant: {class_preds_count.get(1, 0)}")
    return running_loss / len(dataloader.dataset), accuracy, balanced_acc, f1

def gradcam_on_valbags(model, val_dataset, device, n=config.gradcam_n, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(config.results_dir, config.gradcam_dir)
    import matplotlib.cm as cm
    model.eval()
    selected = [i for i, p in enumerate(val_dataset) if p['bag_size'] <= config.min_bag_size_for_aug]
    selected = selected[:n]
    for idx in selected:
        data = val_dataset[idx]
        images = data['images'].unsqueeze(0).to(device)
        label = data['label'].item()
        patient_id = data['patient_id']
        feats = []
        def hook_fn(module, inp, outp):
            feats.append(outp)
        h = model.backbone.conv_head.register_forward_hook(hook_fn) if hasattr(model.backbone, 'conv_head') else model.backbone.global_pool.register_forward_hook(hook_fn)
        logits, attn, _ = model([data['images'].to(device)])
        h.remove()
        pred = torch.argmax(logits, dim=1).item()
        model.zero_grad()
        score = logits[0, pred]
        score.backward(retain_graph=True)
        for i in range(min(data['bag_size'], 5)):
            img = data['images'][i].cpu().numpy().transpose(1,2,0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            heat = attn[0][i].item()
            plt.figure(figsize=(3,3))
            plt.imshow(img)
            plt.title(f"{patient_id} | True:{label} Pred:{pred} Attn:{heat:.2f}")
            plt.axis('off')
            plt.gca().imshow(np.ones_like(img)*heat, alpha=0.3, cmap='jet')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{patient_id.replace('/','_')}_ins{i}_attn.png"))
            plt.close()

def validate_epoch(model, dataloader, criterion, device, val_dataset=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
            images_list = batch['images']
            labels = batch['label'].to(device)
            logits, _, _ = model(images_list)
            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    cm = confusion_matrix(all_labels, all_preds)
    benign_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
    malignant_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)
    custom_metric = 0.7 * benign_recall + 0.3 * malignant_recall
    print(f"Val Loss: {running_loss / len(dataloader.dataset):.4f} | "
          f"Accuracy: {accuracy:.4f} | Balanced Acc: {balanced_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Benign recall: {benign_recall:.4f} | Malignant recall: {malignant_recall:.4f} | Custom metric (ckpt): {custom_metric:.4f}")
    if val_dataset is not None:
        gradcam_on_valbags(model, val_dataset, device)
    results = {
        'loss': running_loss / len(dataloader.dataset),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'auc': auc,
        'cm': cm,
        'benign_recall': benign_recall,
        'malignant_recall': malignant_recall,
        'custom_metric': custom_metric
    }
    return results

def train_fold_with_progressive_unfreezing(train_patients, val_patients, fold, config):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}/{config.num_folds}")
    print(f"{'='*50}")

    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(0.1),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = MILDataset(
        train_patients, transform=train_transform,
        training=True, max_bag_size=config.max_bag_size,
        min_bag_size_for_aug=config.min_bag_size_for_aug
    )
    val_dataset = MILDataset(
        val_patients, transform=val_transform,
        training=False, max_bag_size=config.max_bag_size,
        min_bag_size_for_aug=config.min_bag_size_for_aug
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=mil_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        collate_fn=mil_collate_fn, pin_memory=True
    )
    model = GatedAttentionMIL(config).to(config.device)
    print(f"Model parameters: Total={sum(p.numel() for p in model.parameters()):,}, "
          f"Trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    weights = torch.tensor(config.class_weights, dtype=torch.float32).to(config.device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    total_steps = max(1, len(train_loader) * config.num_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=total_steps
    )

    patience = config.patience
    max_unfreeze_steps = len(model.backbone.blocks)
    unfreeze_idx = -2
    patience_counter = 0
    best_custom_metric = -float('inf')
    best_model_weights = None
    unfrozen_blocks = []
    history = {
        'train_loss': [], 'train_acc': [], 'train_balanced_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_balanced_acc': [], 'val_f1': [], 'val_auc': [],
        'val_benign_recall': [], 'val_malignant_recall': [], 'val_custom_metric': []
    }

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs} | Unfrozen blocks: {unfrozen_blocks}")
        train_loss, train_acc, train_bal_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, config.device, scheduler
        )
        val_metrics = validate_epoch(
            model, val_loader, criterion, config.device, 
            val_dataset=val_dataset if epoch % 10 == 0 else None
        )
        val_bal_acc = val_metrics['balanced_accuracy']
        custom_metric = val_metrics['custom_metric']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_balanced_acc'].append(train_bal_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_benign_recall'].append(val_metrics['benign_recall'])
        history['val_malignant_recall'].append(val_metrics['malignant_recall'])
        history['val_custom_metric'].append(val_metrics['custom_metric'])

        if custom_metric > best_custom_metric:
            best_custom_metric = custom_metric
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"New best custom metric (benign recall prioritized): {best_custom_metric:.4f}")
            model_path = os.path.join(models_dir, f'best_model_fold{fold}.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                if abs(unfreeze_idx) <= max_unfreeze_steps:
                    print(f"Unfreezing block: {unfreeze_idx}")
                    for param in model.backbone.blocks[unfreeze_idx].parameters():
                        param.requires_grad = True
                    unfrozen_blocks.append(unfreeze_idx)
                    optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=config.learning_rate, weight_decay=config.weight_decay
                    )
                    remaining_epochs = max(1, config.num_epochs - epoch)
                    total_steps = max(1, len(train_loader) * remaining_epochs)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=config.learning_rate, total_steps=total_steps
                    )
                    unfreeze_idx -= 1
                    patience_counter = 0
                    print(f"Block {unfreeze_idx+1} unfrozen. Continuing training.")
                else:
                    print("All backbone blocks are unfrozen or patience exhausted. Early stopping.")
                    break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    plot_training_history(history, fold)
    return history

def plot_training_history(history, fold):
    try:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(history['train_balanced_acc'], label='Train Balanced Acc')
        axes[1, 0].plot(history['val_balanced_acc'], label='Val Balanced Acc')
        axes[1, 0].set_title('Balanced Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Balanced Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(history['train_f1'], label='Train F1')
        axes[1, 1].plot(history['val_f1'], label='Val F1')
        axes[1, 1].plot(history['val_auc'], label='Val AUC')
        axes[1, 1].set_title('F1 and AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[2, 0].plot(history['val_benign_recall'], label='Val Benign Recall')
        axes[2, 0].plot(history['val_malignant_recall'], label='Val Malignant Recall')
        axes[2, 0].set_title('Class Recalls')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Recall')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        axes[2, 1].plot(history['val_custom_metric'], label='Val Custom Metric')
        axes[2, 1].set_title('Custom Model Selection Metric')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Custom Metric')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'history_fold{fold}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting history: {e}")

def main_5fold_cv():
    print("="*60)
    print("STARTING 5-FOLD CROSS-VALIDATION")
    print("="*60)

    config_path = os.path.join(config.results_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for attr in dir(config):
            if not attr.startswith('__') and attr != 'device':
                value = getattr(config, attr)
                f.write(f"{attr} = {value}\n")

    patient_list = prepare_patient_list(config.data_root)
    labels = [p['label'] for p in patient_list]
    print(f"Total patients: {len(patient_list)}")
    print(f"Class distribution - Benign: {sum(l==0 for l in labels)}, Malignant: {sum(l==1 for l in labels)}")

    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    all_histories = []

    cv_data = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_list, labels), 1):
        for i in train_idx:
            cv_data.append({'patient_id': patient_list[i]['id'], 'fold': fold_idx, 'split': 'train'})
        for i in val_idx:
            cv_data.append({'patient_id': patient_list[i]['id'], 'fold': fold_idx, 'split': 'val'})

    cv_df = pd.DataFrame(cv_data)
    cv_df.to_csv(os.path.join(config.results_dir, 'cv_splits.csv'), index=False)

    start_fold = 1
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_list, labels), 1):
        if fold < start_fold:
            continue
        print(f"Starting FOLD {fold}")
        train_patients = [patient_list[i] for i in train_idx]
        val_patients = [patient_list[i] for i in val_idx]

        fold_gradcam_dir = os.path.join(gradcam_dir, f'fold{fold}')
        if not os.path.exists(fold_gradcam_dir):
            os.makedirs(fold_gradcam_dir)

        history = train_fold_with_progressive_unfreezing(train_patients, val_patients, fold, config)
        all_histories.append(history)

        fold_history_df = pd.DataFrame(history)
        fold_history_df.to_csv(os.path.join(config.results_dir, f'history_fold{fold}.csv'), index=False)

        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    if all_histories:
        summary_data = []
        for fold, history in enumerate(all_histories, start=start_fold):
            max_idx = np.argmax(history['val_custom_metric'])
            summary_data.append({
                'fold': fold,
                'best_epoch': max_idx + 1,
                'val_balanced_acc': history['val_balanced_acc'][max_idx],
                'val_benign_recall': history['val_benign_recall'][max_idx],
                'val_malignant_recall': history['val_malignant_recall'][max_idx],
                'val_custom_metric': history['val_custom_metric'][max_idx],
                'val_acc': history['val_acc'][max_idx],
                'val_f1': history['val_f1'][max_idx],
                'val_auc': history['val_auc'][max_idx],
                'val_loss': history['val_loss'][max_idx]
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.loc['mean'] = summary_df.mean(numeric_only=True)
        summary_df.loc['std'] = summary_df.std(numeric_only=True)
        summary_df.to_csv(os.path.join(config.results_dir, 'summary_results.csv'), index=True)
        print("\nSummary of best results:")
        print(f"Mean balanced accuracy: {summary_df.loc['mean', 'val_balanced_acc']:.4f} ± {summary_df.loc['std', 'val_balanced_acc']:.4f}")
        print(f"Mean benign recall: {summary_df.loc['mean', 'val_benign_recall']:.4f} ± {summary_df.loc['std', 'val_benign_recall']:.4f}")
        print(f"Mean custom metric: {summary_df.loc['mean', 'val_custom_metric']:.4f} ± {summary_df.loc['std', 'val_custom_metric']:.4f}")
        print(f"Mean AUC: {summary_df.loc['mean', 'val_auc']:.4f} ± {summary_df.loc['std', 'val_auc']:.4f}")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting at: {start_time}")
    print(f"Results will be saved to: {config.results_dir}")

    main_5fold_cv()

    end_time = datetime.now()
    print(f"Completed at: {end_time}")
    print(f"Total time: {end_time - start_time}")

    with open(os.path.join(config.results_dir, 'execution_time.txt'), 'w') as f:
        f.write(f"Started: {start_time}\n")
        f.write(f"Completed: {end_time}\n")
        f.write(f"Total execution time: {end_time - start_time}")