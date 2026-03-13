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
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = unit

        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)

class WideDeep(nn.Module):

    def __init__(
        self,
        feature_columns,
        embedding_dim=16,
        dnn_units=[64,32],
        dropout=0.2
    ):
        super().__init__()

        self.feature_columns = feature_columns
        self.embedding_dim = embedding_dim

        # ----------------------
        # embedding layers
        # ----------------------

        self.embeddings = nn.ModuleDict({
            fc["name"]: nn.Embedding(fc["vocab_size"], embedding_dim)
            for fc in feature_columns
        })

        # ----------------------
        # linear weights
        # ----------------------

        self.linear_embeddings = nn.ModuleDict({
            fc["name"]: nn.Embedding(fc["vocab_size"], 1)
            for fc in feature_columns
        })

        # ----------------------
        # cross feature tables
        # ----------------------

        self.cross_embeddings = nn.ModuleDict()

        for i in range(len(feature_columns)):
            for j in range(i+1, len(feature_columns)):

                fc_i = feature_columns[i]
                fc_j = feature_columns[j]

                name = f"{fc_i['name']}_{fc_j['name']}"

                cross_vocab_size = fc_i["vocab_size"] * fc_j["vocab_size"]

                self.cross_embeddings[name] = nn.Embedding(
                    cross_vocab_size,
                    1
                )

        # ----------------------
        # DNN
        # ----------------------

        input_dim = embedding_dim * len(feature_columns)

        self.dnn = DNN(input_dim, dnn_units, dropout)

        self.deep_output = nn.Linear(dnn_units[-1],1)

    def forward(self, x):

        # x: dict(feature_name -> tensor)

        batch_size = list(x.values())[0].shape[0]

        # ----------------------
        # linear logits
        # ----------------------

        linear_logits = []

        for name, tensor in x.items():

            weight = self.linear_embeddings[name](tensor)

            linear_logits.append(weight)

        linear_logit = torch.sum(torch.cat(linear_logits,dim=1),dim=1,keepdim=True)

        # ----------------------
        # cross logits
        # ----------------------

        cross_logits = []

        feature_names = list(x.keys())

        for i in range(len(feature_names)):
            for j in range(i+1,len(feature_names)):

                fi = feature_names[i]
                fj = feature_names[j]

                vi = x[fi]
                vj = x[fj]

                vocab_j = self.feature_columns[j]["vocab_size"]

                combined_index = vi * vocab_j + vj

                emb = self.cross_embeddings[f"{fi}_{fj}"](combined_index)

                cross_logits.append(emb)

        cross_logit = torch.sum(torch.cat(cross_logits,dim=1),dim=1,keepdim=True)

        # ----------------------
        # deep part
        # ----------------------

        embed_list = []

        for name,tensor in x.items():

            emb = self.embeddings[name](tensor)

            embed_list.append(emb)

        deep_input = torch.cat(embed_list,dim=1)

        deep_out = self.dnn(deep_input)

        deep_logit = self.deep_output(deep_out)

        # ----------------------
        # wide + deep
        # ----------------------

        logits = linear_logit + cross_logit + deep_logit

        output = torch.sigmoid(logits)

        return output