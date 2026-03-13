import torch
from model import DIN

# -------------------------------------------------------
# Build a DIN model
# -------------------------------------------------------
# user_feature_sizes: 2 user features
#   - user_id with vocabulary size 10000
#   - age group with vocabulary size 10
# item_feature_sizes: 2 item features
#   - item_id with vocabulary size 50000
#   - category with vocabulary size 200
model = DIN(
    user_feature_sizes=[10000, 10],
    item_feature_sizes=[50000, 200],
    embedding_dim=32,
    attention_units=[64, 16],
    dnn_units=[256, 128, 64],
    dropout=0.2
)

batch_size = 4
seq_len    = 10   # length of each user's behaviour sequence

# -------------------------------------------------------
# Simulate a mini-batch
# -------------------------------------------------------

# User profile features: [batch_size, num_user_features]
# Each column's values must stay within its own vocabulary size
user_inputs = torch.stack([
    torch.randint(0, 10000, (batch_size,)),  # user_id  (vocab 10000)
    torch.randint(0, 10,    (batch_size,)),  # age group (vocab 10)
], dim=1)

# Candidate item features: [batch_size, num_item_features]
item_inputs = torch.stack([
    torch.randint(0, 50000, (batch_size,)),  # item_id   (vocab 50000)
    torch.randint(0, 200,   (batch_size,)),  # category  (vocab 200)
], dim=1)

# Behaviour sequence features: [batch_size, seq_len, num_item_features]
history_inputs = torch.stack([
    torch.randint(0, 50000, (batch_size, seq_len)),  # item_id   (vocab 50000)
    torch.randint(0, 200,   (batch_size, seq_len)),  # category  (vocab 200)
], dim=2)

# Mask: 1 for real interactions, 0 for padding
# Here the first sample has 3 padding steps at the end
history_mask           = torch.ones(batch_size, seq_len)
history_mask[0, -3:]   = 0   # last 3 positions of the first user are padding

# -------------------------------------------------------
# Forward pass
# -------------------------------------------------------
output = model(user_inputs, item_inputs, history_inputs, history_mask)

print("Output shape:", output.shape)   # Expected: torch.Size([4, 1])
print("Predicted CTR:\n", output)
