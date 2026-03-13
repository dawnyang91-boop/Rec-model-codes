import torch
import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, num_user_features, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_user_features, embedding_dim)

    def forward(self, user_feature_ids):

        # user_feature_ids
        # shape = [batch_size, num_user_features]

        user_embeddings = self.embedding(user_feature_ids)

        # [batch_size, num_features, emb_dim]

        user_embedding_sum = torch.sum(user_embeddings, dim=1)

        # [batch_size, emb_dim]

        ones = torch.ones(user_embedding_sum.size(0), 1, device=user_embedding_sum.device)

        user_vector = torch.cat([ones, user_embedding_sum], dim=1)

        # [batch_size, emb_dim + 1]

        return user_vector
    
class ItemTower(nn.Module):

    def __init__(self, num_item_features, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_item_features, embedding_dim)

        # 一阶线性项
        self.linear = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, item_feature_ids):

        # item_feature_ids
        # [batch_size, num_item_features]

        item_embeddings = self.embedding(item_feature_ids)

        # [batch_size, num_features, emb_dim]

        # ∑(v_t * x_t)

        item_embedding_sum = torch.sum(item_embeddings, dim=1)

        # -------------------------
        # 一阶线性项
        # -------------------------

        item_linear = self.linear(item_embedding_sum)

        # [batch_size,1]

        # -------------------------
        # FM 二阶交叉
        # -------------------------

        sum_square = item_embedding_sum ** 2

        # (∑v)^2

        square = item_embeddings ** 2

        squared_sum = torch.sum(square, dim=1)

        # ∑(v^2)

        fm_vector = sum_square - squared_sum

        fm_interaction = 0.5 * torch.sum(fm_vector, dim=1, keepdim=True)

        # [batch_size,1]

        # -------------------------
        # 合并一阶+FM
        # -------------------------

        first_term = item_linear + fm_interaction

        # -------------------------
        # 拼接最终 item vector
        # -------------------------

        item_vector = torch.cat([first_term, item_embedding_sum], dim=1)

        # [batch_size, emb_dim + 1]

        return item_vector
    
class TwoTowerFM(nn.Module):

    def __init__(self,
                 num_user_features,
                 num_item_features,
                 embedding_dim):

        super().__init__()

        self.user_tower = UserTower(num_user_features, embedding_dim)

        self.item_tower = ItemTower(num_item_features, embedding_dim)

    def forward(self, user_features, item_features):

        user_vector = self.user_tower(user_features)

        item_vector = self.item_tower(item_features)

        # 点积

        score = torch.sum(user_vector * item_vector, dim=1)

        return score