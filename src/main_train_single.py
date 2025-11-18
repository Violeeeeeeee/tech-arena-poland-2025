import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import Subset, DataLoader
import pandas as pd

from utils.config_loader import load_config, parse_main_args
from modules.pooling_archs import DensePoolingHead
from utils.preprocessing import SentencePairDatasetWithEmbeddings, collate_fn_with_embeddings
from modules.train import train_one_epoch
import torch.nn.functional as F
from utils.split import data_split

@torch.no_grad()
def evaluate(model, loader, config, device):
    model.eval()
    all_preds, all_targets_teacher = [], []
    with torch.no_grad():
        for e1, e2, s1, s2, labs in loader:
            e1, e2, s1, s2, labs = e1.to(device), e2.to(device), s1.to(device), s2.to(device), labs.to(device)

            emb1 = model(e1)
            emb2 = model(e2)

            student_cos = F.cosine_similarity(emb1, emb2)
            teacher_cos = F.cosine_similarity(s1, s2)

            all_preds.append(student_cos.cpu())
            all_targets_teacher.append(teacher_cos.cpu())

        all_preds = torch.cat(all_preds)
        all_targets_teacher = torch.cat(all_targets_teacher)

        df = pd.DataFrame({'preds': all_preds.numpy(), 'targets': all_targets_teacher.numpy()})
        pearson = torch.corrcoef(torch.stack([all_preds, all_targets_teacher]))[0, 1].item()

        spearman = 0.0
        try:
            spearman = df.corr(method='spearman')['preds']['targets']
        except Exception as e:
            print(f"Warning: Could not calculate Spearman corr. {e}")
    return {"val_pearson": pearson, "val_spearman": spearman}

def main():
    args = parse_main_args()
    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])
    mlflow.set_experiment(config['experiment_name'])
    print(f"--- Starting Experiment (Single Run): {config['experiment_name']} ---")

    data = torch.load(config['data_path'], weights_only=False)
    dataset = SentencePairDatasetWithEmbeddings(data["embeddings1"], data["embeddings2"], data["sentences1"], data["sentences2"], data["labels"])

    # --- Train/Validation Split ---
    train_idx, val_idx = data_split(dataset, config['train_split_ratio'], config['seed'])
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn_with_embeddings)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn_with_embeddings)

    print(f"Data loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # --- Створення ОДИНОЧНОГО запуску MLflow ---
    run_name = f"Run_{config['loss']['type']}_{config['model']['output_dim']}D"
    with mlflow.start_run(run_name=run_name) as run:
        # Логуємо всі параметри з конфігу
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_params(config['loss'])
        mlflow.log_param("data_path", config['data_path'])

        model = DensePoolingHead(
            input_dim=768,
            output_dim=config['model']['output_dim'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation']
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )

        sample_batch = next(iter(val_loader))
        # e1_sample - це [B, L, D] тензор токенів
        e1_sample = sample_batch[0].to(device)

        best_spearman = -float('inf')

        print("--- Starting Training ---")
        for epoch in range(config['epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
            val_metrics = evaluate(model, val_loader, config, device)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Pearson: {val_metrics['val_pearson']:.4f} | Val Spearman: {val_metrics['val_spearman']:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_pearson", val_metrics['val_pearson'], step=epoch)
            mlflow.log_metric("val_spearman", val_metrics['val_spearman'], step=epoch)

            # Збереження найкращої моделі
            if val_metrics['val_spearman'] > best_spearman:
                best_spearman = val_metrics['val_spearman']
                # 1. Створюємо зразок (NumPy на CPU, як того вимагає MLflow)
                example_input_numpy = e1_sample.cpu().numpy()
                # 2. ТИМЧАСОВО переміщуємо модель на CPU
                model.cpu()
                # 3. Логуємо модель (тепер і модель на CPU, і зразок - NumPy/CPU)
                try:
                    mlflow.pytorch.log_model(
                        model,
                        name="best_model",
                        input_example=example_input_numpy
                    )
                except Exception as e:
                    print(f"Warning: MLflow model logging failed. {e}")
                # 4. ОБОВ'ЯЗКОВО ПОВЕРТАЄМО модель на CUDA для продовження тренування!
                model.to(device)
                mlflow.log_metric("best_spearman_epoch", epoch, step=epoch)

        mlflow.log_metric("final_best_spearman", best_spearman)
        print("\n--- Training Finished ---")
        print(f"Best Spearman: {best_spearman:.4f}")

if __name__ == "__main__":
    main()
