# DIN: Deep Interest Network
# Paper: "Deep Interest Network for Click-Through Rate Prediction" (KDD 2018, Alibaba)
#
# Core idea:
#   Traditional recommendation models compress a user's entire behavior history
#   into a single fixed-length vector, losing fine-grained interest signals.
#   DIN introduces a LOCAL ACTIVATION UNIT (attention mechanism) that adaptively
#   weights each historical behavior according to its relevance to the target item.
#
# Architecture overview:
#   1. Embed user profile features (age, gender, ...)
#   2. Embed target item features (item_id, category, ...)
#   3. Embed the user's historical behavior sequence with the SAME item embeddings
#   4. Attention: for each history item, compute a score against the target item
#   5. Weighted sum of history embeddings -> activated interest vector
#   6. Concatenate [user_profile | target_item | interest_vector] -> DNN -> CTR

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """
    Generic fully-connected feed-forward network used as the top-level predictor.
    Each hidden layer follows the pattern: Linear -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for unit in hidden_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = unit

        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)


class LocalActivationUnit(nn.Module):
    """
    The attention scoring module (a.k.a. Local Activation Unit) in DIN.

    For each (target_item, history_item) pair the network outputs a scalar
    attention weight that reflects how relevant that history item is to the
    current candidate.

    Input features for each pair (all of dimension item_emb_dim):
        h      - history item embedding
        t      - target item embedding (broadcast to sequence length)
        h - t  - element-wise difference  (captures mismatch signal)
        h * t  - element-wise product     (captures interaction signal)

    Concatenating the four parts gives a 4 * item_emb_dim vector that is
    passed through a small MLP to produce a single score.
    """

    def __init__(self, item_emb_dim, attention_units=[64, 16], dropout=0.0):
        super().__init__()

        # The four parts each have size item_emb_dim, so input is 4x
        input_dim = item_emb_dim * 4

        layers = []
        prev_dim = input_dim

        for unit in attention_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = unit

        # Output one scalar score per (target, history_item) pair
        layers.append(nn.Linear(prev_dim, 1))

        self.attention_net = nn.Sequential(*layers)

    def forward(self, target_emb, history_emb):
        """
        Args:
            target_emb:  [batch_size, item_emb_dim]
            history_emb: [batch_size, seq_len, item_emb_dim]

        Returns:
            attention_scores: [batch_size, seq_len, 1]  (raw logits, not normalised)
        """

        seq_len = history_emb.size(1)

        # Expand target to every position in the sequence: [batch_size, seq_len, item_emb_dim]
        target_expand = target_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Difference and element-wise product capture richer interaction signals
        diff    = history_emb - target_expand   # [batch_size, seq_len, item_emb_dim]
        product = history_emb * target_expand   # [batch_size, seq_len, item_emb_dim]

        # Concatenate all four parts along the feature dimension
        # Result: [batch_size, seq_len, 4 * item_emb_dim]
        concat = torch.cat([history_emb, target_expand, diff, product], dim=-1)

        # Pass through the MLP: [batch_size, seq_len, 1]
        attention_scores = self.attention_net(concat)

        return attention_scores


class DIN(nn.Module):
    """
    Deep Interest Network (DIN)

    Parameters
    ----------
    user_feature_sizes : list[int]
        Vocabulary size for each user profile feature
        e.g. [user_id_vocab, age_vocab, gender_vocab]

    item_feature_sizes : list[int]
        Vocabulary size for each item feature (shared by target item and history)
        e.g. [item_id_vocab, category_vocab]

    embedding_dim : int
        Dimension of each feature embedding vector

    attention_units : list[int]
        Hidden layer sizes of the Local Activation Unit MLP

    dnn_units : list[int]
        Hidden layer sizes of the top-level prediction DNN

    dropout : float
        Dropout rate applied in both the attention MLP and the prediction DNN
    """

    def __init__(
        self,
        user_feature_sizes,
        item_feature_sizes,
        embedding_dim=32,
        attention_units=[64, 16],
        dnn_units=[256, 128, 64],
        dropout=0.2
    ):
        super().__init__()

        # ----------------------
        # user profile embeddings
        # ----------------------

        # Each user feature (user_id, age, gender, ...) gets its own lookup table
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in user_feature_sizes
        ])

        # ----------------------
        # item embeddings
        # (shared between target item and history items)
        # ----------------------

        # Using the SAME embedding table for target and history ensures that
        # the dot-product-like interaction in the attention unit is meaningful
        self.item_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in item_feature_sizes
        ])

        # Dimension of a single item's concatenated embedding vector
        item_emb_dim = embedding_dim * len(item_feature_sizes)

        # ----------------------
        # local activation unit (attention)
        # ----------------------

        self.attention = LocalActivationUnit(
            item_emb_dim=item_emb_dim,
            attention_units=attention_units,
            dropout=dropout
        )

        # ----------------------
        # prediction DNN
        # ----------------------

        # DNN input = user_profile_emb || target_item_emb || interest_emb
        user_profile_dim = embedding_dim * len(user_feature_sizes)
        dnn_input_dim    = user_profile_dim + item_emb_dim + item_emb_dim

        self.dnn          = DNN(dnn_input_dim, dnn_units, dropout)
        self.output_layer = nn.Linear(dnn_units[-1], 1)

    def forward(self, user_inputs, item_inputs, history_inputs, history_mask=None):
        """
        Args
        ----
        user_inputs    : [batch_size, num_user_features]            integer indices
        item_inputs    : [batch_size, num_item_features]            integer indices
        history_inputs : [batch_size, seq_len, num_item_features]   integer indices
        history_mask   : [batch_size, seq_len]  optional boolean/float mask;
                         1 for real behaviours, 0 for padding

        Returns
        -------
        output : [batch_size, 1]  predicted click-through probability in (0, 1)
        """

        # ----------------------
        # user profile embedding
        # ----------------------

        user_emb_list = []
        for i, emb in enumerate(self.user_embeddings):
            # Look up the i-th user feature: [batch_size, embedding_dim]
            user_emb_list.append(emb(user_inputs[:, i]))

        # Concatenate all user feature embeddings: [batch_size, user_profile_dim]
        user_profile_emb = torch.cat(user_emb_list, dim=1)

        # ----------------------
        # target item embedding
        # ----------------------

        target_emb_list = []
        for i, emb in enumerate(self.item_embeddings):
            # Look up the i-th item feature for the candidate: [batch_size, embedding_dim]
            target_emb_list.append(emb(item_inputs[:, i]))

        # Concatenate: [batch_size, item_emb_dim]
        target_emb = torch.cat(target_emb_list, dim=1)

        # ----------------------
        # history behaviour embedding
        # ----------------------

        history_emb_list = []
        for i, emb in enumerate(self.item_embeddings):
            # Look up the i-th item feature over the whole sequence:
            # [batch_size, seq_len, embedding_dim]
            history_emb_list.append(emb(history_inputs[:, :, i]))

        # Concatenate along the feature dimension: [batch_size, seq_len, item_emb_dim]
        history_emb = torch.cat(history_emb_list, dim=-1)

        # ----------------------
        # attention-based pooling
        # ----------------------

        # Raw attention scores: [batch_size, seq_len, 1]
        attention_scores = self.attention(target_emb, history_emb)

        if history_mask is not None:
            # Expand mask to match attention_scores shape: [batch_size, seq_len, 1]
            mask = history_mask.unsqueeze(-1).float()
            # Add -1e9 to padding positions so they get ~0 weight after softmax;
            # valid positions (mask=1) are unaffected because (1-1)*(-1e9) = 0
            attention_scores = attention_scores + (1.0 - mask) * (-1e9)

        # Normalise attention weights over the sequence: [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum over history: [batch_size, item_emb_dim]
        # This is the "activated" user interest vector conditioned on the target item
        interest_emb = torch.sum(history_emb * attention_weights, dim=1)

        # ----------------------
        # concatenate & predict
        # ----------------------

        # Concatenate all three representations: [batch_size, dnn_input_dim]
        dnn_input = torch.cat([user_profile_emb, target_emb, interest_emb], dim=1)

        dnn_output = self.dnn(dnn_input)                    # [batch_size, dnn_units[-1]]

        logit  = self.output_layer(dnn_output)              # [batch_size, 1]
        output = torch.sigmoid(logit)                       # map to (0, 1) probability

        return output
