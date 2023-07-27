import torch
from torch import nn
import torch.nn.functional as F

class Prototype(nn.Module):
    def __init__(self, embedding_size, hidden_size, label_size, device):
        super(Prototype, self).__init__()
        self.hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.prototype_layer = nn.Linear(hidden_size, embedding_size)
        self.label_size = label_size
        self.device = device

    def forward(self, x, batch, global_prototype_tensor):
        prototype_embeddings = torch.zeros(self.label_size, x.shape[-1], device=self.device) 
        anti_prototype_embeddings_list = []  # List to store each anti-prototype

        for i in range(0, self.label_size):
            label_embeddings_list = []
            anti_label_embeddings_list = []
            for idx, labels in enumerate(batch['label_list']):
                if i in labels:
                    label_embeddings_list.append(x[idx, i, :])
                else:
                    anti_label_embeddings_list.append(x[idx, i, :])

            if len(label_embeddings_list) > 0:
                label_embeddings = torch.stack(label_embeddings_list)
                global_prototype = global_prototype_tensor[i]  # index for prototype
                avg_embedding = 0.5 * label_embeddings.mean(dim=0) + 0.5 * global_prototype
                avg_embedding = F.relu(self.hidden_layer(avg_embedding))
                prototype_embedding = self.prototype_layer(avg_embedding)
                prototype_embeddings[i, :] = prototype_embedding  # Directly update the tensor
                # print("prototype_embeddings",prototype_embeddings.shape)

            if len(anti_label_embeddings_list) > 0:
                anti_label_embeddings = torch.stack(anti_label_embeddings_list)
                global_anti_prototype = global_prototype_tensor[-1]  # index for anti-prototype
                avg_anti_embedding = 0.5 * anti_label_embeddings.mean(dim=0) + 0.5 * global_anti_prototype
                avg_anti_embedding = F.relu(self.hidden_layer(avg_anti_embedding))
                anti_prototype_embedding = self.prototype_layer(avg_anti_embedding)
                # print("anti_prototype_embedding",anti_prototype_embedding.shape)
                anti_prototype_embeddings_list.append(anti_prototype_embedding)

        # Average all the anti-prototypes to get single anti-prototype
        anti_prototype = torch.stack(anti_prototype_embeddings_list).mean(dim=0)
        prototype_embeddings = torch.cat([prototype_embeddings, anti_prototype.unsqueeze(0)], dim=0)
        # print("prototype_embeddings_final",prototype_embeddings.shape)
        return prototype_embeddings
