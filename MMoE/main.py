import torch
from model import MMoE

# -------------------------------------------------------
# Build an MMoE model for 2-task learning
#   Task 1: Click-Through Rate (CTR) prediction
#   Task 2: Conversion Rate  (CVR) prediction
# -------------------------------------------------------
# feature_sizes: 3 sparse input features
#   - user_id   with vocabulary size 10 000
#   - item_id   with vocabulary size 50 000
#   - category  with vocabulary size 200
model = MMoE(
    feature_sizes=[10000, 50000, 200],
    embedding_dim=16,
    num_experts=4,
    expert_units=[128, 64],
    num_tasks=2,
    tower_units=[32],
    dropout=0.2
)

batch_size = 4

# -------------------------------------------------------
# Simulate a mini-batch
# -------------------------------------------------------

# Feature inputs: [batch_size, num_features]
# Each column's values must stay within its respective vocabulary size
feature_inputs = torch.stack([
    torch.randint(0, 10000, (batch_size,)),   # user_id  (vocab 10000)
    torch.randint(0, 50000, (batch_size,)),   # item_id  (vocab 50000)
    torch.randint(0, 200,   (batch_size,)),   # category (vocab 200)
], dim=1)

# -------------------------------------------------------
# Forward pass
# -------------------------------------------------------

outputs = model(feature_inputs)

ctr_output, cvr_output = outputs

print("CTR output shape:", ctr_output.shape)   # Expected: torch.Size([4, 1])
print("CVR output shape:", cvr_output.shape)   # Expected: torch.Size([4, 1])
print("Predicted CTR:\n", ctr_output)
print("Predicted CVR:\n", cvr_output)
