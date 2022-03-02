import torch
import torch_geometric

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv

from torch_cluster import radius_graph, knn_graph

from torch_scatter import scatter_add

class GraphMETNetwork_dyn(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=2, k=10, activation_function = nn.ELU()):
        super(GraphMETNetwork_dyn, self).__init__()
        
        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        self.embed_pv = nn.Embedding(4, hidden_dim//4)
        
        self.embed_continuous = nn.Sequential(nn.Linear(continuous_dim,hidden_dim//2),
                                              activation_function,
                                              #nn.BatchNorm1d(hidden_dim//2) # uncomment if it starts overtraining
                                             )

        self.embed_categorical = nn.Sequential(nn.Linear(3*hidden_dim//4,hidden_dim//2),
                                               activation_function,                                               
                                               #nn.BatchNorm1d(hidden_dim//2)
                                              )

        self.encode_all = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        activation_function
                                       )
        self.bn_all = nn.BatchNorm1d(hidden_dim)
        
        self.conv_continuous = nn.ModuleList()        
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim))
            self.conv_continuous.append(nn.ModuleList())
            self.conv_continuous[-1].append(EdgeConv(nn=mesg).jittable())
            self.conv_continuous[-1].append(nn.BatchNorm1d(hidden_dim))
        

        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    activation_function,
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )
        
        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

        self.k = k
        
    
    def forward(self, x, edge_index, batch):
        
        x_cont = x[:,:7]
        x_cat = x[:,8:].long()

        emb_cont = self.embed_continuous(x_cont)        
        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)
        emb_pv = self.embed_pv(x_cat[:, 2])

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg, emb_pv], dim=1))
        
        emb = self.bn_all(self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)))
        
        # graph convolution for continuous variables
        for co_conv in self.conv_continuous:
            emb = co_conv[1](co_conv[0](emb, knn_graph(emb, k=self.k, batch=batch, loop=True)))

        out = self.output(emb)
        
        return out.squeeze(-1)
    


class GraphMETNetwork_fix_emb(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=2, activation_function = nn.ELU()):
        super(GraphMETNetwork_fix_emb, self).__init__()
        
        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        self.embed_pv = nn.Embedding(4, hidden_dim//4)
        
        self.embed_continuous = nn.Sequential(nn.Linear(continuous_dim,hidden_dim//2),
                                              activation_function,
                                              #nn.BatchNorm1d(hidden_dim//2) # uncomment if it starts overtraining
                                             )

        self.embed_categorical = nn.Sequential(nn.Linear(3*hidden_dim//4,hidden_dim//2),
                                               activation_function,                                               
                                               #nn.BatchNorm1d(hidden_dim//2)
                                              )

        self.encode_all = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        activation_function,
                                       )
        self.bn_all = nn.BatchNorm1d(hidden_dim)
        

        self.conv_continuous = nn.ModuleList()
        for i in range(conv_depth):
            self.conv_continuous.append(GCNConv(hidden_dim, hidden_dim))
        
        """
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    activation_function,
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )
        """
        self.output1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                            activation_function)
        self.output2 = nn.Sequential(nn.Linear(hidden_dim//2, output_dim))

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]
        

    def forward(self, x, edge_index, batch):
        
        x_cont = x[:,:7]
        x_cat = x[:,8:].long()

        emb_cont = self.embed_continuous(x_cont)        
        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)
        emb_pv = self.embed_pv(x_cat[:, 2])

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg, emb_pv], dim=1))
        
        emb = self.bn_all(self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)))
        
        # graph convolution for continuous variables
        for co_conv in self.conv_continuous:
            emb = co_conv(emb, edge_index)
            emb = F.relu(emb)
            emb = F.dropout(emb, training=self.training)
                
        out = self.output1(emb)
        #print(out.shape)
        out = self.output2(out)

        #dscb_input = torch.mul(torch.sum(emb,1),torch.squeeze(torch.transpose(out,0,1)))
        dscb_input = scatter_add(torch.transpose(emb,1,0)*torch.squeeze(out), batch)
        dscb_input = torch.transpose(dscb_input, 0,1)
        #dscb_input = torch.reshape(dscb_input,(-1,))
        #print(dscb_input.size())
        dscb_layer = nn.Sequential(nn.Linear(32,32), 
                                    nn.BatchNorm1d(32,32), nn.ELU(),
                                    nn.Linear(32,32), 
                                    nn.BatchNorm1d(32,32), nn.ELU(),
                                    nn.Linear(32,32), 
                                    nn.BatchNorm1d(32,32), nn.ELU(),
                                    nn.Linear(32,5), nn.Softplus()).to('cuda')
        dscb_out = torch.transpose(dscb_layer(dscb_input),0,1)
        #print(dscb_out)

        return out.squeeze(-1), dscb_out

"""
class dscb(nn.Module):
    def __init__(self, input_dim, output_dim=5):
        super(dscb, self).__init__()
        dscb_layer = nn.Sequential(nn.Linear(self.dscb_input.dim(),5), nn.ELU())


    def get_dscb_parameters(self):

        dscb-layer = nn.Sequential(nn.Linear(self.dscb_input.dim(),5), nn.ELU())
        out = dscb-layer(self.dscb_input)
        return out.squeeze(-1)
"""



class GraphMETNetwork_fix_noemb(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=2, activation_function = nn.ELU()):
        super(GraphMETNetwork_fix_noemb, self).__init__()
        dim = 11

        self.conv_continuous = nn.ModuleList()
        self.conv_continuous.append(GCNConv(dim, hidden_dim))
        for i in range(conv_depth-1):
            self.conv_continuous.append(GCNConv(hidden_dim, hidden_dim))

        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    activation_function,
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )

    #def forward(self, x_cont, x_cat, edge_index, batch):
    def forward(self, x, edge_index, batch):
        
        for co_conv in self.conv_continuous:
            x = co_conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.output(x)

        return x.squeeze(-1)