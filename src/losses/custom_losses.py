import torch
import torch.nn.functional as F

def cosent_loss_fn(emb1: torch.Tensor, emb2: torch.Tensor, hard_labels: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """
    CoSENT (Cosine Sentence) Loss.
    Використовує hard_labels для ранжування, незалежно від scale.
    Це модифікація, де emb1 і emb2 - це вже ембеддінги, а labels - це scores (0-5).
    """
    y_true = (hard_labels[:, None] < hard_labels[None, :]).float() # [Batch, Batch]
    student_cos_matrix = torch.matmul(F.normalize(emb1, p=2, dim=-1), F.normalize(emb2, p=2, dim=-1).T)
    y_pred = student_cos_matrix.diag() * tau # diagonal elements
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    res = torch.logsumexp(y_pred, dim=0)
    return res

# --- Factory-function for unification ---

def calculate_loss(
    student_emb1: torch.Tensor,
    student_emb2: torch.Tensor,
    teacher_emb1: torch.Tensor,
    teacher_emb2: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    alpha: float = 0.7,
    tau: float = 20.0 # for CoSENT
) -> torch.Tensor:

    # Prediction: Student Cosine [B]
    student_cos = F.cosine_similarity(student_emb1, student_emb2)

    # Target: Teacher Cosine (Soft Target) [B]
    teacher_cos = F.cosine_similarity(teacher_emb1, teacher_emb2)

    if loss_name == "hard_mse":
        # Чистий Supervised: student_cos vs. normalized hard labels
        return F.mse_loss(student_cos, labels)

    elif loss_name == "soft_mse":
        # Чистий Distillation (Teacher Logits/Score): student_cos vs. teacher_cos
        return F.mse_loss(student_cos, teacher_cos)

    elif loss_name == "hybrid_kd":
        loss_distill = F.mse_loss(student_cos, teacher_cos)
        loss_label = F.mse_loss(student_cos, labels)
        return alpha * loss_distill + (1 - alpha) * loss_label

    elif loss_name == "cosent_rank":
        # CoSENT (навчає ранжуванню)
        # Примітка: CoSENT не використовує Soft Target напряму.
        # Навчаємо Student Embeddings за ранжуванням, використовуючи Hard Labels (0-5).
        return cosent_loss_fn(student_emb1, student_emb2, labels, tau)

    else:
        # Можливість розширення:
        # if loss_name == "new_idea_loss": return new_idea_loss_function(...)
        raise ValueError(f"Unknown loss function: {loss_name}")
