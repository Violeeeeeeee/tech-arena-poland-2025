import torch
import os
import time
import warnings # Додано для придушення попереджень
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import mlflow
from mlflow.models.signature import infer_signature

# Ваші імпорти
from utils.config_loader import load_config, parse_main_args
from modules.pooling_archs import DensePoolingHead
from utils.preprocessing import SentencePairDatasetWithEmbeddings, collate_fn_with_embeddings
from modules.train import train_one_epoch, evaluate
from utils.split import kfold_split

# --- Фільтрація набридливих попереджень ---
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
# Ігноруємо специфічне попередження про версії pip
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

def main():
    args = parse_main_args()
    config = load_config(args.config)
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    print(f"--- Starting Experiment: {config['experiment_name']} ---")

    # --- MLflow Setup ---
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(config['experiment_name'])

    # --- Load Data ---
    # Ігноруємо warning про weights_only
    data = torch.load(config['data_path'], weights_only=False)
    dataset = SentencePairDatasetWithEmbeddings(data["embeddings1"], data["embeddings2"], data["sentences1"], data["sentences2"], data["labels"])

    folds = kfold_split(dataset, k=config['kfolds'], seed=config['seed'])
    fold_results = []

    # --- Parent Run ---
    with mlflow.start_run(run_name="ParentRun_KFold") as parent_run:
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_params(config['loss'])

        for fold, (train_idx, val_idx) in enumerate(folds):
            # --- Fold Run ---
            with mlflow.start_run(nested=True, run_name=f"Fold {fold + 1}") as child_run:
                start_time = time.time()
                print(f"\n--- Fold {fold + 1}/{config['kfolds']} ---")
                mlflow.log_param("fold_number", fold + 1)

                # Loaders
                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)
                train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn_with_embeddings)
                val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn_with_embeddings)

                # Model setup
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

                # Variables to store BEST weights (in memory only)
                best_state_pearson = None
                best_val_pearson = float("-inf")
                best_state_spearman = None
                best_val_spearman = float("-inf")
                best_state_mse = None
                best_val_mse = float("inf")

                # === TRAINING LOOP ===
                for epoch in range(config['epochs']):
                    epoch_start = time.time()
                    train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
                    val_metrics = evaluate(model, val_loader, config, device)

                    # 1. Зберігаємо ваги в RAM, якщо результат кращий
                    if val_metrics['val_spearman'] > best_val_spearman:
                        best_val_spearman = val_metrics['val_spearman']
                        best_state_spearman = model.state_dict().copy()

                    if val_metrics['val_pearson'] > best_val_pearson:
                        best_val_pearson = val_metrics['val_pearson']
                        best_state_pearson = model.state_dict().copy()

                    if val_metrics['val_mse'] < best_val_mse:
                        best_val_mse = val_metrics['val_mse']
                        best_state_mse = model.state_dict().copy()

                    # 2. Логуємо ТІЛЬКИ числа (метрики), НЕ модель
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_mse", val_metrics['val_mse'], step=epoch)
                    mlflow.log_metric("val_spearman", val_metrics['val_spearman'], step=epoch)
                    mlflow.log_metric("val_pearson", val_metrics['val_pearson'], step=epoch)

                    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Sp: {val_metrics['val_spearman']:.4f} | Pe: {val_metrics['val_pearson']:.4f} | Time: {time.time()-epoch_start:.1f}s")

                # === END OF EPOCHS ===

                # Тільки ТЕПЕР, один раз на фолд, ми логуємо модель
                fold_results.append(best_val_spearman)

                # Підготовка для чистого логування
                sample_batch = next(iter(val_loader))
                e1_sample = sample_batch[0].cpu().numpy() # Вхід (numpy)

                # Створюємо "чисті" requirements щоб уникнути warning про версії
                pip_reqs = [f"torch=={torch.__version__.split('+')[0]}", "numpy", "pandas"]

                # Функція для безпечного логування
                def safe_log_model(state_dict, artifact_name):
                    if state_dict:
                        model.load_state_dict(state_dict)
                        model.cpu() # Переносимо на CPU

                        # Генеруємо підпис (Signature)
                        try:
                            with torch.no_grad():
                                output = model(torch.from_numpy(e1_sample))
                            signature = infer_signature(e1_sample, output.numpy())
                        except Exception:
                            signature = None

                        mlflow.pytorch.log_model(
                            model,
                            artifact_path=artifact_name,
                            input_example=e1_sample,
                            signature=signature,
                            pip_requirements=pip_reqs
                        )
                        model.to(device) # Повертаємо на GPU, якщо треба

                print("Logging best models to MLflow...")
                safe_log_model(best_state_spearman, "best_model_spearman")
                safe_log_model(best_state_mse, "best_model_mse")
                safe_log_model(best_state_pearson, "best_model_pearson")

                mlflow.log_metric("Final_Best_Spearman", best_val_spearman)
                print(f"Fold Finished. Best Spearman: {best_val_spearman:.4f}")

    # Summary
    avg_spearman = sum(fold_results) / len(fold_results)
    mlflow.log_metric("avg_kfold_spearman", avg_spearman)
    print(f"\n=== Experiment Finished. Avg Spearman: {avg_spearman:.4f} ===")

if __name__ == '__main__':
    main()
