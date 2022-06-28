import torch

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import EdgeConv, GCNConv

from torch_cluster import knn_graph

func_dict = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
}


class GraphMET_GCNConv(nn.Module):
    def __init__(
        self,
        continuous_dim,
        cat_dim,
        output_dim=1,
        hidden_dim=32,
        conv_depth=2,
        activation_function="elu",
    ):
        super(GraphMET_GCNConv, self).__init__()

        self.n_cont = continuous_dim
        self.n_cat = cat_dim

        self.embed_charge = nn.Embedding(3, hidden_dim // 4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim // 4)
        self.embed_frompv = nn.Embedding(4, hidden_dim // 4)

        self.mlp_categorical = nn.Sequential(
            nn.Linear(3 * hidden_dim // 4, hidden_dim // 2),
            func_dict[activation_function],
            # nn.BatchNorm1d(hidden_dim // 2)
        )

        self.mlp_continuous = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim // 2),
            func_dict[activation_function],
            # nn.BatchNorm1d(hidden_dim // 2) # uncomment if it starts overtraining
        )

        self.mlp_all = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            func_dict[activation_function],
        )

        self.batchnorm_all = nn.BatchNorm1d(hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(conv_depth):
            self.convs.append(GCNConv(hidden_dim, hidden_dim).jittable())

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            func_dict[activation_function],
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x, edge_index, batch):

        end_cont = self.n_cont
        x_cont = x[:, :end_cont]
        begin_cat = x.size()[1] - self.n_cat # categorical variables are at the end
        x_cat = x[:, begin_cat:].long()

        emb_cont = self.mlp_continuous(x_cont)

        emb_charge = self.embed_charge(x_cat[:, 1] + 1)
        emb_frompv = self.embed_frompv(x_cat[:, 2])
        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(
                pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap
            )
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.mlp_categorical(torch.cat([emb_charge, emb_pdg, emb_frompv], dim=1))

        emb = self.batchnorm_all(self.mlp_all(torch.cat([emb_cat, emb_cont], dim=1)))

        # graph convolution
        for conv in self.convs:
            emb = conv(emb, edge_index)
            emb = F.relu(emb)
            emb = F.dropout(emb, training=self.training)

        out = self.output(emb)

        return out.squeeze(-1)


class GraphMET_EdgeConv(nn.Module):
    def __init__(
        self,
        graph_info,
        cont_dim,
        cat_dim,
        output_dim=1,
        hidden_dim=32,
        conv_depth=2,
        activation_function="elu",
        output_activation="sigmoid",
    ):
        super(GraphMET_EdgeConv, self).__init__()

        self.n_cont = cont_dim
        self.n_cat = cat_dim

        self.embed_charge = nn.Embedding(3, hidden_dim // 4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim // 4)
        self.embed_frompv = nn.Embedding(4, hidden_dim // 4)

        self.mlp_categorical = nn.Sequential(
            nn.Linear(3 * hidden_dim // 4, hidden_dim // 2),
            func_dict[activation_function],
            # nn.BatchNorm1d(hidden_dim // 2)
        )

        self.mlp_continuous = nn.Sequential(
            nn.Linear(cont_dim, hidden_dim // 2),
            func_dict[activation_function],
            # nn.BatchNorm1d(hidden_dim // 2) # uncomment if it starts overtraining
        )

        self.mlp_all = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            func_dict[activation_function],
        )

        self.batchnorm_all = nn.BatchNorm1d(hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(conv_depth):
            mesg = nn.Sequential(
                nn.Linear(2 * hidden_dim, graph_info["mesg_dim"]),
                func_dict[activation_function],
                nn.Linear(graph_info["mesg_dim"], hidden_dim),
                func_dict[activation_function],
            )
            self.convs.append(nn.ModuleList())
            self.convs[-1].append(EdgeConv(nn=mesg).jittable())
            # self.conv[-1].append(nn.BatchNorm1d(hidden_dim))

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            func_dict[activation_function],
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.output_activation = func_dict[output_activation]

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x, edge_index, batch):

        end_cont = self.n_cont
        x_cont = x[:, :end_cont]
        begin_cat = x.size()[1] - self.n_cat # categorical variables are at the end
        x_cat = x[:, begin_cat:].long()

        emb_cont = self.mlp_continuous(x_cont)

        emb_charge = self.embed_charge(x_cat[:, 1] + 1)
        emb_frompv = self.embed_frompv(x_cat[:, 2])
        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(
                pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap
            )
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.mlp_categorical(torch.cat([emb_charge, emb_pdg, emb_frompv], dim=1))
        emb = self.batchnorm_all(self.mlp_all(torch.cat([emb_cat, emb_cont], dim=1)))

        # graph convolution
        for conv in self.convs:
            emb = emb + conv[0](emb, edge_index)

        out = self.output(emb)
        out = out.squeeze(-1)

        return self.output_activation(out)


class GraphMET_dynamicEdgeConv(nn.Module):
    def __init__(
        self,
        graph_info,
        cont_dim,
        cat_dim,
        output_dim=1,
        hidden_dim=32,
        conv_depth=2,
        activation_function="elu",
        output_activation="sigmoid",
    ):
        super(GraphMET_dynamicEdgeConv, self).__init__()

        self.n_cont = cont_dim
        self.n_cat = cat_dim
        self.k = graph_info["k"]
        self.loop = graph_info["loop"]

        self.embed_charge = nn.Embedding(3, hidden_dim // 4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim // 4)
        self.embed_frompv = nn.Embedding(4, hidden_dim // 4)

        self.mlp_categorical = nn.Sequential(
            nn.Linear(3 * hidden_dim // 4, hidden_dim // 2),
            func_dict[activation_function],
            # nn.BatchNorm1d(hidden_dim // 2)
        )

        self.mlp_continuous = nn.Sequential(
            nn.Linear(cont_dim, hidden_dim // 2),
            func_dict[activation_function],
            # nn.BatchNorm1d(hidden_dim // 2) # uncomment if it starts overtraining
        )

        self.mlp_all = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), func_dict[activation_function]
        )
        self.batchnorm_all = nn.BatchNorm1d(hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2 * hidden_dim, graph_info["mesg_dim"]),
                func_dict[activation_function],
                nn.Linear(graph_info["mesg_dim"], hidden_dim),
                func_dict[activation_function],)
            self.convs.append(nn.ModuleList())
            self.convs[-1].append(EdgeConv(nn=mesg).jittable())
            self.convs[-1].append(nn.BatchNorm1d(hidden_dim))

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            func_dict[activation_function],
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.output_activation = func_dict[output_activation]

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x, edge_index, batch):

        end_cont = self.n_cont
        x_cont = x[:, :end_cont]
        begin_cat = x.size()[1] - self.n_cat # categorical variables are at the end
        x_cat = x[:, begin_cat:].long()

        emb_cont = self.mlp_continuous(x_cont)

        emb_charge = self.embed_charge(x_cat[:, 1] + 1)
        emb_frompv = self.embed_frompv(x_cat[:, 2])
        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(
                pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap
            )
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.mlp_categorical(torch.cat([emb_charge, emb_pdg, emb_frompv], dim=1))

        emb = self.batchnorm_all(self.mlp_all(torch.cat([emb_cat, emb_cont], dim=1)))

        # graph convolution
        for conv in self.convs:
            emb = conv[1](
                conv[0](emb, knn_graph(emb, k=self.k, batch=batch, loop=self.loop))
            )

        out = self.output(emb)
        out = out.squeeze(-1)

        return self.output_activation(out)