# MMoE: Multi-gate Mixture-of-Experts
# Paper: "Modeling Task Relationships in Multi-task Learning with
#         Multi-gate Mixture-of-Experts" (KDD 2018, Google)
#
# Core idea:
#   In multi-task learning, hard-parameter sharing (a single shared DNN trunk)
#   can hurt performance when tasks conflict or are only loosely related.
#   MMoE replaces the single shared trunk with K independent EXPERT networks.
#   Each task then has its own lightweight GATE network that learns a soft
#   (weighted) mixture over all K experts, letting the model automatically
#   decide how much to share or specialise per task.
#
# Architecture overview:
#   1. Embed sparse input features into dense vectors
#   2. Feed the concatenated embedding into K expert MLPs in parallel
#   3. For each task t: gate_t(x) → softmax weights over K experts
#   4. For each task t: weighted sum of expert outputs → task tower → sigmoid output

import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    A single expert network: a standard MLP with ReLU activations.

    All K experts share the same architecture but have independent parameters,
    allowing them to specialise in different aspects of the input.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (concatenated embedding vector)
    expert_units : list[int]
        Hidden layer sizes, e.g. [128, 64]
    dropout : float
        Dropout probability applied after each hidden layer
    """

    def __init__(self, input_dim, expert_units, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for unit in expert_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = unit

        self.net = nn.Sequential(*layers)

        # Expose output dimension so the caller can wire up downstream layers
        self.output_dim = prev_dim

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, expert_units[-1]]
        """
        return self.net(x)


class Gate(nn.Module):
    """
    Gating network for one task.

    Takes the same input vector as the experts and produces a probability
    distribution over K experts, controlling each expert's contribution.

    A linear projection followed by softmax is sufficient; keeping the gate
    lightweight avoids overfitting the mixing weights.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (same as Expert.input_dim)
    num_experts : int
        Number of experts to choose among
    """

    def __init__(self, input_dim, num_experts):
        super().__init__()

        # Linear layer maps input to a score for each expert
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            gate_weights: [batch_size, num_experts]
                          softmax distribution — sums to 1 across experts
        """
        # Raw scores → probability distribution over experts
        return torch.softmax(self.gate(x), dim=1)


class TaskTower(nn.Module):
    """
    Task-specific tower network.

    Receives the gated mixture of expert outputs for one task and maps it
    to a final scalar prediction (logit).  A sigmoid is applied outside this
    module so that the raw logit can also be used with BCEWithLogitsLoss.

    Parameters
    ----------
    input_dim : int
        Dimension of the mixed expert representation
    tower_units : list[int]
        Hidden layer sizes, e.g. [32]
    dropout : float
        Dropout probability applied after each hidden layer
    """

    def __init__(self, input_dim, tower_units, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for unit in tower_units:
            layers.append(nn.Linear(prev_dim, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = unit

        # Final projection to a single scalar logit
        layers.append(nn.Linear(prev_dim, 1))

        self.tower = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            logit: [batch_size, 1]  (un-normalised score)
        """
        return self.tower(x)


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE)

    Designed for multi-task learning in recommendation systems.  Typical
    usage pairs two binary tasks such as CTR (click-through rate) and CVR
    (conversion rate) prediction on the same feature set.

    Parameters
    ----------
    feature_sizes : list[int]
        Vocabulary size for each sparse input feature.
        e.g. [user_id_vocab, item_id_vocab, category_vocab]

    embedding_dim : int
        Embedding dimension shared across all feature lookup tables

    num_experts : int
        Number of expert networks (K in the paper; typically 4–8)

    expert_units : list[int]
        Hidden layer sizes for each expert MLP, e.g. [128, 64]

    num_tasks : int
        Number of tasks (output heads); one gate + one tower per task

    tower_units : list[int]
        Hidden layer sizes for each task-specific tower, e.g. [32]

    dropout : float
        Dropout rate applied inside experts and task towers
    """

    def __init__(
        self,
        feature_sizes,
        embedding_dim=16,
        num_experts=4,
        expert_units=None,
        num_tasks=2,
        tower_units=None,
        dropout=0.2
    ):
        super().__init__()

        # Use safe defaults for mutable arguments
        if expert_units is None:
            expert_units = [128, 64]
        if tower_units is None:
            tower_units = [32]

        # -------------------------
        # feature embedding tables
        # -------------------------

        # Each sparse feature (user_id, item_id, …) gets its own lookup table
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in feature_sizes
        ])

        # Dimension of the concatenated embedding vector fed to experts / gates
        input_dim = embedding_dim * len(feature_sizes)

        # -------------------------
        # expert networks (shared across all tasks)
        # -------------------------

        # K experts with identical architecture but independent parameters
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_units, dropout)
            for _ in range(num_experts)
        ])

        # Output dimension of each expert (= last hidden size)
        expert_output_dim = self.experts[0].output_dim

        # -------------------------
        # gating networks (one per task)
        # -------------------------

        # Each gate sees the same raw input and produces weights over K experts
        self.gates = nn.ModuleList([
            Gate(input_dim, num_experts)
            for _ in range(num_tasks)
        ])

        # -------------------------
        # task-specific towers (one per task)
        # -------------------------

        self.towers = nn.ModuleList([
            TaskTower(expert_output_dim, tower_units, dropout)
            for _ in range(num_tasks)
        ])

    def forward(self, feature_inputs):
        """
        Args
        ----
        feature_inputs : [batch_size, num_features]
            Integer indices for each sparse feature column

        Returns
        -------
        outputs : list of tensors, one per task
            Each tensor has shape [batch_size, 1] and contains a predicted
            probability in (0, 1) after applying sigmoid
        """

        # -------------------------
        # embed input features
        # -------------------------

        embed_list = []
        for i, emb in enumerate(self.embeddings):
            # Look up the i-th feature for every sample: [batch_size, embedding_dim]
            embed_list.append(emb(feature_inputs[:, i]))

        # Concatenate all feature embeddings: [batch_size, input_dim]
        x = torch.cat(embed_list, dim=1)

        # -------------------------
        # compute expert outputs
        # -------------------------

        # Each expert independently transforms the same input representation
        # expert_outputs: list of K tensors, each [batch_size, expert_output_dim]
        expert_outputs = [expert(x) for expert in self.experts]

        # Stack along a new dimension: [batch_size, num_experts, expert_output_dim]
        expert_stack = torch.stack(expert_outputs, dim=1)

        # -------------------------
        # compute task-specific mixtures and predictions
        # -------------------------

        outputs = []
        for gate, tower in zip(self.gates, self.towers):

            # Gate weights for this task: [batch_size, num_experts]
            gate_weights = gate(x)

            # Expand for broadcasting over the expert_output_dim axis:
            # [batch_size, num_experts, 1]
            gate_weights = gate_weights.unsqueeze(-1)

            # Weighted sum of expert outputs: [batch_size, expert_output_dim]
            # Each expert's representation is scaled by the gate's assigned weight
            mixed = torch.sum(expert_stack * gate_weights, dim=1)

            # Task tower maps the mixture to a scalar logit: [batch_size, 1]
            logit = tower(mixed)

            # Sigmoid maps the logit to a probability in (0, 1)
            output = torch.sigmoid(logit)
            outputs.append(output)

        return outputs
