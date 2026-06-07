import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from mamba_ssm import Mamba

from torch.utils.data import DataLoader
from utils.SSTDataLoader import create_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM = 640
PREDICT_DAY = 15
DATA_DIR = "your dir"
LABEL_DIR = "your dir"

class GCBlock(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.relu = nn.ReLU()
        self.GCNconv = GCNConv(30, 30)
        self.fc1 = nn.Linear(num_nodes, num_nodes)
        self.fc_i = nn.Linear(30, 1)
        self.fc_j = nn.Linear(30, 1)
        
    def From_A_to_Edge_index(self, possibility_A):
        A = torch.where(possibility_A > 0.1, possibility_A, torch.zeros_like(possibility_A))
        indices = torch.where(A > 0.1)
        rows, cols = indices
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = A[rows, cols]
        return A, edge_index, edge_weight

    def min_max_normalize(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        if tensor_max - tensor_min == 0:
            return torch.zeros_like(tensor)
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized
    
    def gaussian_similarity_matrix(self,A, sigma=None):
        """
        similarity(i,j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        """
        diff = A.unsqueeze(1) - A.unsqueeze(0)
        squared_dist = torch.sum(diff ** 2, dim=2) # [NumNodes, NumNodes]

        if sigma is None:
            sigma = torch.median(squared_dist.sqrt())
        similarity = torch.exp(-squared_dist / (2 * sigma ** 2))

        return similarity

    def SNN(self, z_sequence, A):
        mu = 0.3
        vi = 0
        vj = 0
        C_i = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        C_j = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        
        for T in range(3):
            # Charging
            i_sequence = self.fc_i(z_sequence)
            j_sequence = self.fc_j(z_sequence)
            mi = vi + i_sequence
            mj = vj + j_sequence
            
            # Fire
            ci = torch.where(mi >= mu, 1, 0)
            cj = torch.where(mj >= mu, 1, 0)
            
            # Reset
            vi = mi - mu*ci
            vj = mj - mu*cj
            
            C_i += ci
            C_j += cj

        C_i = C_i / 3
        C_j = C_j / 3
        C_i = C_i.expand(-1, self.num_nodes) # [NumNodes, NumNodes]
        C_j = C_j.t().expand(self.num_nodes, -1)
        A_cij = torch.where(A != 0, C_i + C_j, torch.zeros_like(A))
        
        return A_cij

    def GAT(self, A_cij):
        row_sum_sqrt = torch.sqrt(A_cij.sum(dim=1))
        col_sum_sqrt = torch.sqrt(A_cij.sum(dim=0))
        denominator = torch.outer(row_sum_sqrt, col_sum_sqrt)
        A_aij = torch.where(A_cij != 0, A_cij / denominator, torch.zeros_like(A_cij))
        return A_aij

    def forward(self, data):
        # data[bs, 30, NumNodes]
        temperature = data * 0.01

        node_feature = []
        for i in range(data.shape[0]):
            t_feature = temperature[i, :, :] # [30, NumNodes]
            t_feature = t_feature.t()

            possibility_A = self.gaussian_similarity_matrix(t_feature) # [NumNodes, NumNodes]
            possibility_A = self.fc1(possibility_A)
            possibility_A = self.relu(possibility_A)
            possibility_A = self.min_max_normalize(possibility_A)
            
            A, edge_index, edge_weight = self.From_A_to_Edge_index(possibility_A)

            diff = t_feature[:, 1:]-t_feature[:, :-1]
            diff =torch.cat([torch.zeros(self.num_nodes, 1, device=device), diff], dim=1)
            z_sequence = torch.where(diff > 0, 1, torch.zeros_like(diff))
            
            A_cij = self.SNN(z_sequence, A)
            A_aij = self.GAT(A_cij)
            A2, edge_index2, edge_weight2 = self.From_A_to_Edge_index(A_aij)

            x = t_feature # [NumNodes, 30]
            x = self.GCNconv(x=x, edge_index=edge_index2, edge_weight=edge_weight2)
            x = self.relu(x) # [NumNodes, 30]

            x = x.t() # [30, NumNodes]
            node_feature.append(x)

        node_feature = torch.stack(node_feature) # [bs, 30, NumNodes]
        return node_feature

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
    
    def forward(self, x):
        x = self.mamba(x)
        return x

class AOG(nn.Module):
    def __init__(self, num_nodes, dim, num_layers=2, predict_day=15):
        super().__init__()
        self.gcblock = GCBlock(num_nodes=num_nodes)
        self.layers_fore = nn.ModuleList([MambaBlock(dim) for _ in range(num_layers)])
        self.layers_back = nn.ModuleList([MambaBlock(dim) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(30, predict_day)
    def forward(self, x_fore):
        # x[bs, 30, NumNodes]
        x_fore = self.gcblock(x_fore)
        x_back = torch.flip(x_fore, dims = [1])

        for layer in self.layers_fore:
            x_fore = layer(x_fore)
            x_fore = self.relu(x_fore)

        for layer in self.layers_back:
            x_back = layer(x_back)
            x_back = self.relu(x_back)

        x_back = torch.flip(x_back, dims = [1])
        x = x_fore + x_back # [bs, 30, NumNodes]
        x = x.transpose(1, 2) # [bs, NumNodes, 30]
        x = self.fc(x) # [bs, NumNodes, PREDICT_DAY]
        x = x.transpose(1, 2) # [bs, PREDICT_DAY, NumNodes]
        return x

if __name__ == '__main__':

    
    train_loader, val_loader, test_loader, indices = create_dataloaders(DATA_DIR, LABEL_DIR, batch_size=64)

    model = AOG(num_nodes=NUM, dim=NUM, num_layers=2, predict_day=PREDICT_DAY)
    model.to(device)

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        print(output.shape)
        print(label.shape)
        break


    # test_data = torch.rand(1, 15, 648)
    # test_data = test_data.to(device)
    # test_model = MambaBlock(dim = 648)
    # test_model.to(device)
    # test_output = test_model(test_data)
    # print(test_output.shape)