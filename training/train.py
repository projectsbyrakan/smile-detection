import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import mlflow
from training.data_loading import mydata, train_transform, test_transform
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report


# ===== TRAINING FUNCTION =====
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)


# ===== EVALUATION FUNCTION =====
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels


# ===== MAIN =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Smile Classifier')
    parser.add_argument('--model', type=str, required=True,
                       choices=['efficientnet', 'mobilenet'],
                       help='Model to train')
    args = parser.parse_args()

    # ===== LOAD MODEL CONFIG =====
    if args.model == 'efficientnet':
        from models.efficientnet import get_model, config
    elif args.model == 'mobilenet':
        from models.mobilenet import get_model, config

    # ===== MLFLOW =====
    mlflow.set_experiment("smile-classifier")
    
    with mlflow.start_run(run_name=config["run_name"]):
        
        params = {
            "architecture": config["architecture"],
            "dataset": "smile_dataset_cropped",
            "stage1_epochs": 5,
            "stage1_lr": 1e-3,
            "stage2_epochs": 10,
            "stage2_backbone_lr": 1e-5,
            "stage2_head_lr": 1e-4,
            "batch_size": 32,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss_weighted",
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
        }
        mlflow.log_params(params)

        # ===== DATA =====
        df = pd.read_csv('data/image_with_label.csv')
        dataset_size = len(df)
        indices = list(range(dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)

        split_train = int(0.70 * len(indices))
        split_val = int(0.85 * len(indices))
        train_indices = indices[:split_train]
        val_indices = indices[split_train:split_val]
        test_indices = indices[split_val:]

        train_dataset = mydata('data/image_with_label.csv', 'cropped_faces', transform=train_transform)
        val_dataset = mydata('data/image_with_label.csv', 'cropped_faces', transform=test_transform)
        test_dataset = mydata('data/image_with_label.csv', 'cropped_faces', transform=test_transform)

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=params["batch_size"], shuffle=False, num_workers=4)

        print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

        # ===== MODEL =====
        model = get_model()

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        model = model.to(device)

        pos_weight = torch.tensor([3000 / 7000]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        print(f"Using device: {device}")
        print(f"Model: {config['architecture']}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        mlflow.log_param("device", str(device))

        # ===== STAGE 1: FREEZE BACKBONE =====
        print("=" * 50)
        print("STAGE 1: Training classifier head (backbone frozen)")
        print("=" * 50)

        for param in model.features.parameters():
            param.requires_grad = False

        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=params["stage1_lr"])

        for epoch in range(params["stage1_epochs"]):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch + 1)

            print(f"Epoch {epoch+1}/{params['stage1_epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # ===== STAGE 2: UNFREEZE =====
        print("=" * 50)
        print("STAGE 2: Fine-tuning entire model (backbone unfrozen)")
        print("=" * 50)

        for param in model.features.parameters():
            param.requires_grad = True

        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': params["stage2_backbone_lr"]},
            {'params': model.classifier.parameters(), 'lr': params["stage2_head_lr"]}
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        best_val_acc = 0
        for epoch in range(params["stage2_epochs"]):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            step = params["stage1_epochs"] + epoch + 1
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=step)

            print(f"Epoch {epoch+1}/{params['stage2_epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), config["save_path"])
                print(f"  → Saved best model (Val Acc: {val_acc:.4f})")

        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # ===== FINAL TEST =====
        print("=" * 50)
        print("FINAL TEST EVALUATION (unseen data)")
        print("=" * 50)

        model.load_state_dict(torch.load(config["save_path"], map_location=device))
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        f1 = f1_score(labels, preds)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=["not_smiling", "smiling"]))

        mlflow.log_metrics({
            "final_test_loss": test_loss,
            "final_test_acc": test_acc,
            "final_f1_score": f1,
        })
        mlflow.log_artifact(config["save_path"])

        print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")