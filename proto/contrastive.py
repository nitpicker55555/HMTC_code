import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MultiLabelContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, label_lists):
        device = embeddings.device

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=2)  # dim=2 as now shape is [batch_size, num_labels, embedding_size]

        batch_size, num_labels, _ = embeddings.shape

        # Create label tensor
        label_tensor = torch.zeros((batch_size, num_labels), dtype=torch.int64).to(device)
        for i, label_list in enumerate(label_lists):
            label_tensor[i, label_list] = 1

        # Compute positive and negative masks
        mask_positive = torch.einsum('bn,bm->bnm', label_tensor, label_tensor)
        mask_negative = 1 - mask_positive

        # Compute logits
        logits = torch.einsum('bne,bme->bnm', [embeddings, embeddings]) / self.temperature
        # print(logits.shape,"logits before max")
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=2, keepdim=True)
        # print(logits_max.shape,"logits_max")
        logits = logits - logits_max.detach()
        # print(logits.shape,"logits after max")
        # Compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(torch.sum(exp_logits * mask_negative, dim=2, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_positive * log_prob).sum(2) / mask_positive.sum(2)
        mean_log_prob_pos = torch.nan_to_num(mean_log_prob_pos)
        # print(mean_log_prob_pos,"mean_log_prob_pos")
        # Loss
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss