import torch
from model import TwoTowerFM

model = TwoTowerFM(
    num_user_features=10000,
    num_item_features=20000,
    embedding_dim=32
)

user_features = torch.randint(0,10000,(4,3))
item_features = torch.randint(0,20000,(4,3))

score = model(user_features,item_features)

print(score.shape)