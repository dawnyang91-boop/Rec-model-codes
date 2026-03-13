import torch
from model import DNN,DSSM

model = DSSM(
    user_feature_sizes=[10000, 3, 10],
    item_feature_sizes=[50000, 200, 100],
    embedding_dim=32,
    dnn_units=[128,64],
    dropout=0.2
)

batch_size = 4

user_inputs = torch.randint(0,100,(batch_size,3))
item_inputs = torch.randint(0,100,(batch_size,3))

score = model(user_inputs,item_inputs)

print(score)