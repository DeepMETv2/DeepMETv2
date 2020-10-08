import torch
import torch_geometric

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv

from torch_cluster import radius_graph, knn_graph

class GraphMETNetwork(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
        
        self.embed_continuous = nn.Sequential(nn.Linear(continuous_dim,hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             # nn.BatchNorm1d(hidden_dim) # uncomment if it starts overtraining
                                            )

        self.embed_categorical = nn.Sequential(nn.Linear(cat_dim,hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(hidden_dim, hidden_dim),
                                               # nn.BatchNorm1d(hidden_dim)
                                              )

        self.conv_continuous = nn.ModuleList()        
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2*hidden_dim, 3*hidden_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(3*hidden_dim//2, hidden_dim),
                                 # nn.BatchNorm1d(hidden_dim)
                                )

            self.conv_continuous.append(
                EdgeConv(nn=mesg).jittable()
                #GCNConv(hidden_dim, hidden_dim).jittable()
            )
            
        self.conv_categorical = nn.ModuleList()        
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2*hidden_dim, 3*hidden_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(3*hidden_dim//2, hidden_dim),
                                 # nn.BatchNorm1d(hidden_dim)
                                )
            self.conv_categorical.append(
                EdgeConv(nn=mesg).jittable()
                #GCNConv(hidden_dim, hidden_dim).jittable()
            )
        
        self.output = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )

    def forward(self, x_cont, x_cat, edge_index, batch):
        emb_cont = self.embed_continuous(x_cont)
        emb_cat = self.embed_categorical(x_cat)
        
        # graph convolution for continuous variables
        for co_conv in self.conv_continuous:
            #emb_cont = co_conv(emb_cont, edge_index)
            emb_cont = emb_cont + co_conv(emb_cont, edge_index)#residual connections on the convolutional layer

        # graph convolution for discrete variables
        for ca_conv in self.conv_categorical:
            #emb_cat = ca_conv(emb_cat, edge_index)
            emb_cat = emb_cat + ca_conv(emb_cat, edge_index)#residual connections on the convolutional layer    
                              
        # concatenate embeddings together to make description of weight inputs
        emb = torch.cat([emb_cont,emb_cat], dim=1)
        
        out = self.output(emb)
        
        return out.squeeze(-1)
