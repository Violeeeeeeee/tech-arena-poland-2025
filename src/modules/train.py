import torch
from losses.custom_losses import calculate_loss
import torch.nn.functional as F
from scipy.stats import spearmanr


def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss = 0
    for e1, e2, s1, s2, labs in loader:
        e1, e2, s1, s2, labs = e1.to(device), e2.to(device), s1.to(device), s2.to(device), labs.to(device)

        optimizer.zero_grad()

        emb1 = model(e1)
        emb2 = model(e2)

        loss = calculate_loss(
            student_emb1=emb1,
            student_emb2=emb2,
            teacher_emb1=s1,
            teacher_emb2=s2,
            labels=labs,
            loss_name=config['loss']['type'],
            alpha=config['loss']['alpha_kd'],
            tau=config['loss']['tau_cosent']
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, config, device, alpha=0.7):
    model.eval()
    all_preds, all_targets = [], []
    total_mse = 0.0

    with torch.no_grad():
        for e1, e2, s1, s2, labs in loader:
            e1, e2, s1, s2, labs = e1.to(device), e2.to(device), s1.to(device), s2.to(device), labs.to(device)

            emb1 = model(e1)
            emb2 = model(e2)

            student_cos = F.cosine_similarity(emb1, emb2)
            teacher_cos = F.cosine_similarity(s1, s2)

            mse = alpha * F.mse_loss(student_cos, teacher_cos) + (1 - alpha) * F.mse_loss(student_cos, labs)
            total_mse += mse.item()

            all_preds.append(student_cos.cpu())
            all_targets.append(teacher_cos.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        avg_mse = total_mse / len(loader)
        spearman = spearmanr(all_preds.numpy(), all_targets.numpy())[0]
        pearson = torch.corrcoef(torch.stack([all_preds, all_targets]))[0,1].item()

    return {"val_mse": avg_mse, "val_pearson": pearson, "val_spearman": spearman}
