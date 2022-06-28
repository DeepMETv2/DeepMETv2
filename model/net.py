"""Defines the neural network, loss function and metrics"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from model.graph_met_networks import (
    GraphMET_GCNConv,
    GraphMET_EdgeConv,
    GraphMET_dynamicEdgeConv,
)


class Net(nn.Module):
    def __init__(self, net_info):
        super(Net, self).__init__()

        if net_info["graph"]["calculation"] == "static":
            if net_info["graph"]["layer"] == "GCNConv":
                self.graphnet = GraphMET_GCNConv(
                    net_info["continuous_dim"],
                    net_info["categorical_dim"],
                    output_dim=net_info["output_dim"],
                    hidden_dim=net_info["hidden_dim"],
                    conv_depth=net_info["conv_depth"],
                    activation_function=net_info["activation_func"],
                )
            elif net_info["graph"]["layer"] == "EdgeConv":
                self.graphnet = GraphMET_EdgeConv(
                    net_info["graph"],
                    net_info["continuous_dim"],
                    net_info["categorical_dim"],
                    output_dim=net_info["output_dim"],
                    hidden_dim=net_info["hidden_dim"],
                    conv_depth=net_info["conv_depth"],
                    activation_function=net_info["activation_func"],
                    output_activation=net_info["output_func"],
                )
            else:
                print("Graph layer does not exist.")
        elif net_info["graph"]["calculation"] == "dynamic":
            self.graphnet = GraphMET_dynamicEdgeConv(
                net_info["continuous_dim"],
                net_info["categorical_dim"],
                net_info["graph"],
                output_dim=net_info["output_dim"],
                hidden_dim=net_info["hidden_dim"],
                conv_depth=net_info["conv_depth"],
                activation_function=net_info["activation_func"],
            )
        else:
            print("Graph not defined.")

    def forward(self, x, edge_index, batch):
        weights = self.graphnet(x, edge_index, batch)
        return weights #torch.sigmoid(weights)


def loss_fn(pred_weights, variables, truth, batch, loss_info):
    # transverse momentum of PF candidates
    px = variables[:, 0]
    py = variables[:, 1]
    # corrected MET by predicted weights
    METx = scatter_add(pred_weights * px, batch)
    METy = scatter_add(pred_weights * py, batch)
    # generator MET
    true_METx = truth[:, 0]
    true_METy = truth[:, 1]

    if loss_info["puppi_match"]:
        tzero = torch.zeros(variables.shape[0]).to("cuda")
        BCE = nn.BCELoss()
        # BCE compares predicted weights to match puppi weights for charged charged particles, for neutrals always zero
        loss = 0.5 * (
            (METx + true_METx) ** 2 + (METy + true_METy) ** 2
        ).mean() + loss_info["bce_factor"] * BCE(
            torch.where(variables[:, 9] == 0, tzero, pred_weights),
            torch.where(variables[:, 9] == 0, tzero, variables[:, 7]),
        )
    else:
        loss = 0.5 * ((METx + true_METx) ** 2 + (METy + true_METy) ** 2).mean()

    return loss


def resolution(pred_weights, variables, truth, batch):
    def getdot(vx, vy):
        return torch.einsum("bi,bi->b", vx, vy)

    def getscale(vx):
        return torch.sqrt(getdot(vx, vx))

    def scalermul(a, v):
        return torch.einsum("b,bi->bi", a, v)

    # generator MET in x/y direction
    genMETx = truth[:, 0]
    genMETy = truth[:, 1]  
    # generator MET
    v_genMET = torch.stack((genMETx, genMETy), dim=1)

    # particle flow MET in x/y direction
    pfMETx = truth[:, 2] 
    pfMETy = truth[:, 3]
    # particle flow MET
    v_pfMET = torch.stack((pfMETx, pfMETy), dim=1)

    # puppi MET in x/y direction
    puppiMETx = truth[:, 4] 
    puppiMETy = truth[:, 5] 
    # puppi MET
    v_puppiMET = torch.stack((puppiMETx, puppiMETy), dim=1)

    has_deepmet = False
    if truth.size()[1] > 6:
        has_deepmet = True
        # DeepMET Response Tune in x/y direction
        deepMETResponse_x = truth[:, 6]
        deepMETResponse_y = truth[:, 7] 
        # DeepMET Response Tune
        v_deepMETResponse = torch.stack((deepMETResponse_x, deepMETResponse_y), dim=1)

        # DeepMET Response Tune in x/y direction
        deepMETResolution_x = truth[:, 8]  
        deepMETResolution_y = truth[:, 9]  
        # DeepMET Resolution Tune
        v_deepMETResolution = torch.stack(
            (deepMETResolution_x, deepMETResolution_y), dim=1
        )

    # transverse momentum of PF candidates 
    px = variables[:, 0]
    py = variables[:, 1]
    # corrected MET by predicted weights
    METx = scatter_add(pred_weights * px, batch)
    METy = scatter_add(pred_weights * py, batch)
    # predicted MET/qT
    v_MET = torch.stack((METx, METy), dim=1)

    def compute(vector):
        response = getdot(vector, v_genMET) / getdot(v_genMET, v_genMET)
        v_paral_predict = scalermul(response, v_genMET)
        u_paral_predict = getscale(v_paral_predict) - getscale(v_genMET)
        v_perp_predict = vector - v_paral_predict
        u_perp_predict = getscale(v_perp_predict)
        return [
            u_perp_predict.cpu().detach().numpy(),
            u_paral_predict.cpu().detach().numpy(),
            response.cpu().detach().numpy(),
        ]

    resolutions = {
        "graphMET": compute(-v_MET),
        "pfMET": compute(v_pfMET),
        "puppiMET": compute(v_puppiMET),
    }
    if has_deepmet:
        resolutions.update(
            {
                "deepMETResponse": compute(v_deepMETResponse),
                "deepMETResolution": compute(v_deepMETResolution),
            }
        )
    return (
        resolutions,
        torch.sqrt(truth[:, 0] ** 2 + truth[:, 1] ** 2).cpu().detach().numpy(),
    )


# maintain all metrics required in this dictionary, these are used in the training and evaluation loops
metrics = {
    "resolution": resolution,
}
