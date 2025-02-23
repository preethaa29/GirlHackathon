import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1️⃣ Create a sample RTL-based graph
G = nx.DiGraph()
G.add_edges_from([
    ("FF1", "AND1"), ("FF2", "AND1"),
    ("AND1", "OR1"), ("FF3", "OR1"),
    ("OR1", "XOR1"), ("FF4", "XOR1"),
    ("XOR1", "DFF_OUT")
])

# 2️⃣ Feature extraction (fan-in, fan-out count)
nodes = list(G.nodes)
X = np.array([[G.in_degree(n), G.out_degree(n)] for n in nodes])  # Feature matrix
y = np.array([3, 3, 2, 3, 2, 1, 4])  # Simulated logic depth

# 3️⃣ Convert to PyTorch Geometric format
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
x_tensor = torch.tensor(X, dtype=torch.float)
data = Data(x=x_tensor, edge_index=edge_index)

# 4️⃣ Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 8)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = GNN()
pred_features = model(data).detach().numpy()  # Extract GNN-based features

# 5️⃣ Train XGBoost to refine predictions
X_train, X_test, y_train, y_test = train_test_split(pred_features, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# 6️⃣ Evaluation
mae = mean_absolute_error(y_test, y_pred)
print(f"Prediction MAE: {mae}")
