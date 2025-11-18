import torch
import os
from datetime import datetime
import time
from torch.utils.data import Subset, DataLoader
from utils.config_loader import load_config, parse_main_args
from modules.pooling_archs import DensePoolingHead
from utils.preprocessing import SentencePairDatasetWithEmbeddings, collate_fn_with_embeddings

from modules.train import train_one_epoch, evaluate
from utils.split import kfold_split

import mlflow

def main():
    args = parse_main_args()
    config = load_config(args.config)
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    print(f"--- Starting Experiment: {config['experiment_name']} ---")
    print(f"Device: {device}, Output Dim: {config['model']['output_dim']}, Loss: {config['loss']['type']}")

    mlflow.end_run()
    mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config['experiment_name'])
    # Print connection information
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")



    # 4. Динамічна Ініціалізація Моделі
    # Ми використовуємо config['pooling_type'] з YAML, припустимо, це "DensePoolingHead"
    try:
        # Для цього потрібно, щоб DensePoolingHead був доступний тут, або
        # використовуємо getattr, якщо хочете динамічно обирати класи:
        # model_class = getattr(import_module("src.modules.pooling_archs"), config['pooling_type'])

        # Наразі, використовуємо прямий імпорт:
        student_model = DensePoolingHead(
            input_dim=768,
            output_dim=config['model']['output_dim'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation']
        ).to(device)

        print(f"Model Initialized: {student_model.__class__.__name__} with {config['model']['output_dim']}D output.")

    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    data = torch.load(config['data_path'])
    dataset = SentencePairDatasetWithEmbeddings(data["embeddings1"], data["embeddings2"], data["sentences1"], data["sentences2"], data["labels"])
    hhh = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_path = config['save_path'] + f"model{hhh}"
    os.makedirs(full_path, exist_ok=True)

    folds = kfold_split(dataset, k=config['kfolds'], seed=config['seed'])
    namee_pearson = []
    namee_spearman = []
    namee_mse = []

    fold_results = []

    with mlflow.start_run(run_name="ParentRun_KFold") as parent_run:
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_params(config['loss'])
        mlflow.log_param("kfolds", config['kfolds'])

        for fold, (train_idx, val_idx) in enumerate(folds):
            with mlflow.start_run(nested=True, run_name=f"Fold {fold + 1}") as child_run:
                # Log the hyperparameters
                # mlflow.log_params({'seed': config['seed']})
                # mlflow.log_params({'epochs': config['epochs']})
                # mlflow.log_params({'output_dim': config['model']['output_dim']})
                # mlflow.log_params({'dropout': config['model']['dropout']})
                # mlflow.log_params({'activation': config['model']['activation']})
                # mlflow.log_params({'batch_size': config['training']['batch_size']})
                # mlflow.log_params({'lr': config['training']['lr']})
                # mlflow.log_params({'weight_decay': config['training']['weight_decay']})
                # mlflow.log_params({'type': config['loss']['type']})
                # mlflow.log_params({'alpha_kd': config['loss']['alpha_kd']})
                # mlflow.log_params({'tau_cosent': config['loss']['tau_cosent']})
                # mlflow.log_params('', config[''][''])
                # mlflow.log_params('', config[''])
                start_time = time.time()
                print(f"\n--- Fold {fold + 1}/{config['kfolds']} ---")
                mlflow.log_param("fold_number", fold + 1)

                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)
                train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn_with_embeddings)
                val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn_with_embeddings)

                sample_batch = next(iter(val_loader))
                e1_sample = sample_batch[0].to(device)

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

                best_state_pearson = None
                best_val_pearson = float("-inf")

                best_state_spearman = None
                best_val_spearman = float("-inf")

                best_state_mse = None
                best_val_mse = float("inf")

                for epoch in range(config['epochs']):
                    start_epoch = time.time()
                    train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
                    val_metrics = evaluate(model, val_loader, config, device)

                    if val_metrics['val_spearman'] > best_val_spearman:
                        best_val_spearman = val_metrics['val_spearman']
                        best_state_spearman = model.state_dict()

                    if val_metrics['val_pearson'] > best_val_pearson:
                        best_val_pearson = val_metrics['val_pearson']
                        best_state_pearson = model.state_dict()

                    if val_metrics['val_mse'] < best_val_mse:
                        best_val_mse = val_metrics['val_mse']
                        best_state_mse = model.state_dict()

                    epoch_elapsed = time.time() - start_epoch
                    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Pearson: {val_metrics['val_pearson']:.4f} | Val Spearman: {val_metrics['val_spearman']:.4f} | Val MSE: {val_metrics['val_mse']:.4f} | Time: {epoch_elapsed:.2f}s")

                    # --- ЛОГУЄМО В MLFLOW ЛИШЕ МЕТРИКИ ---
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_mse", val_metrics['val_mse'], step=epoch)
                    mlflow.log_metric("val_spearman", val_metrics['val_spearman'], step=epoch)
                    mlflow.log_metric("val_pearson", val_metrics['val_pearson'], step=epoch)


                mlflow.log_metric("Final Best MSE", best_val_mse)
                mlflow.log_metric("Final Best Spearman", best_val_spearman)
                mlflow.log_metric("Final Best Pearson", best_val_pearson)
                # mlflow.log_metric("MSE", val_metrics['val_mse'])
                # mlflow.log_metric("Spearman", val_metrics['val_spearman'])
                # mlflow.log_metric("Pearson", val_metrics['val_pearson'])
                mlflow.pytorch.log_model(model, name="model")

                e1_sample_numpy = e1_sample.cpu().numpy()

                if best_state_spearman is not None:
                    model.load_state_dict(best_state_spearman) # Завантажуємо найкращий стан
                    model.cpu() # Переміщуємо на CPU для логування
                    mlflow.pytorch.log_model(
                        model,
                        name="best_model_spearman",
                        input_example=e1_sample_numpy
                    )
                    model.to(device) # Повертаємо на GPU (на випадок, якщо знадобиться)

                # Логуємо найкращу модель (за MSE)
                if best_state_mse is not None:
                    model.load_state_dict(best_state_mse)
                    model.cpu()
                    mlflow.pytorch.log_model(
                        model,
                        name="best_model_mse",
                        input_example=e1_sample_numpy
                    )
                    model.to(device)


                if best_state_pearson is not None:
                    model.load_state_dict(best_state_pearson)
                    model.cpu()
                    mlflow.pytorch.log_model(
                        model,
                        name="best_model_pearson",
                        input_example=e1_sample_numpy
                    )
                    model.to(device)
                # mlflow.log_metric("Best MSE", best_val_mse)
                # mlflow.log_metric("Best Spearman", best_val_spearman)
                # mlflow.log_metric("Best Pearson", best_val_pearson)
                # mlflow.pytorch.log_model(best_state_mse, name="best_model_mse")
                # mlflow.pytorch.log_model(best_state_pearson, name="best_model_pearson")
                # mlflow.pytorch.log_model(best_state_spearman, name="best_model_spearman")

                fold_results.append(best_val_spearman)

                elapsed = time.time() - start_time
                print(
                    f"Time: {elapsed:.2f}s |  best spearman: {best_val_spearman} | best pearson: {best_val_spearman} | best mse: {best_val_mse}"
                )

                not_yet_full_path = full_path + f"/fold_{fold + 1}/"
                os.makedirs(not_yet_full_path, exist_ok=True)

                name_pearson = f"_pearson_{best_val_pearson:.6f}.bin"
                name_spearman = f"_spearman_{best_val_spearman:.6f}.bin"
                name_mse = f"_mse_{best_val_mse:.6f}.bin"

                full_path_pearson = not_yet_full_path + name_pearson
                full_path_spearman = not_yet_full_path + name_spearman
                full_path_mse = not_yet_full_path + name_mse

                namee_spearman.append(name_spearman)
                namee_pearson.append(name_pearson)
                namee_mse.append(name_mse)


                torch.save(best_state_spearman, full_path_spearman)
                torch.save(best_state_pearson, full_path_pearson)
                torch.save(best_state_mse, full_path_mse)

    mlflow.end_run()
    avg_spearman = sum(fold_results) / len(fold_results)
    print("\n--- K-Fold Finished ---")
    print(f"Average Spearman across {config['kfolds']} folds: {avg_spearman:.4f}")


if __name__ == '__main__':
    main()
