import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for unit in hidden_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(nn.BatchNorm1d(unit))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = unit

        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)
    
class DSSM(nn.Module):

    def __init__(
        self,
        user_feature_sizes,
        item_feature_sizes,
        embedding_dim=32,
        dnn_units=[128,64],
        dropout=0.2
    ):
        super().__init__()

        # ---------------------
        # embedding layers
        # ---------------------

        self.user_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in user_feature_sizes
        ])

        self.item_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in item_feature_sizes
        ])

        user_input_dim = embedding_dim * len(user_feature_sizes)
        item_input_dim = embedding_dim * len(item_feature_sizes)

        # ---------------------
        # towers
        # ---------------------

        self.user_dnn = DNN(user_input_dim, dnn_units, dropout)
        self.item_dnn = DNN(item_input_dim, dnn_units, dropout)

    def forward(self, user_inputs, item_inputs):

        # ---------------------
        # user tower
        # ---------------------

        user_embed_list = []

        for i, emb in enumerate(self.user_embeddings):
            user_embed_list.append(emb(user_inputs[:, i]))

        user_feature = torch.cat(user_embed_list, dim=1)

        user_tower = self.user_dnn(user_feature)

        # L2 normalize

        user_embedding = F.normalize(user_tower, p=2, dim=1)

        # ---------------------
        # item tower
        # ---------------------

        item_embed_list = []

        for i, emb in enumerate(self.item_embeddings):
            item_embed_list.append(emb(item_inputs[:, i]))

        item_feature = torch.cat(item_embed_list, dim=1)

        item_tower = self.item_dnn(item_feature)

        item_embedding = F.normalize(item_tower, p=2, dim=1)

        # ---------------------
        # cosine similarity
        # ---------------------

        score = torch.sum(user_embedding * item_embedding, dim=1)

        return score