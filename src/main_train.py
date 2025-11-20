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
from torch.utils.tensorboard import SummaryWriter

def main():
    args = parse_main_args()
    config = load_config(args.config)
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    print(f"--- Starting Experiment: {config['experiment_name']} ---")
    print(f"Device: {device}, Output Dim: {config['model']['output_dim']}, Loss: {config['loss']['type']}")


    try:
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

    for fold, (train_idx, val_idx) in enumerate(folds):
            start_time = time.time()
            print(f"\n--- Fold {fold + 1}/{config['kfolds']} ---")

            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn_with_embeddings)
            val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn_with_embeddings)

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


            fold_results.append(best_val_spearman)

            elapsed = time.time() - start_time
            print(f"Fold Time: {elapsed:.2f}s | Best Spearman: {best_val_spearman:.4f}")

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

    avg_spearman = sum(fold_results) / len(fold_results)
    print("\n--- K-Fold Finished ---")
    print(f"Average Spearman across {config['kfolds']} folds: {avg_spearman:.4f}")

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

if __name__ == '__main__':
    main()
