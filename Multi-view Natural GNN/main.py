from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

# Expected input: one row per spatial unit with ID, area, coordinates,
# a binary class label, and 17 socio-ecological predictors.
ID_COL = "FID"
AREA_COL = "Shape_Area"
LON_COL = "X"
LAT_COL = "Y"
LABEL_COL = "type"

FEATURE_COLUMNS = [
    "elevation",
    "slope",
    "species_richness_I",
    "species_richness_II",
    "cropland_proportion",
    "forest_proportion",
    "grassland_proportion",
    "water_body_proportion",
    "residential_proportion",
    "unused_land_proportion",
    "road_density",
    "NPP",
    "precipitation",
    "temperature",
    "nighttime_light",
    "population_density",
    "NDVI",
]

def load_input_table(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    required = [ID_COL, AREA_COL, LON_COL, LAT_COL, LABEL_COL] + FEATURE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[ID_COL, AREA_COL, LON_COL, LAT_COL, LABEL_COL]).copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[ID_COL] = df[ID_COL].astype(str)
    return df.reset_index(drop=True)


def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return radius * 2 * atan2(sqrt(a), sqrt(1 - a))


def gravitational_weight(area1, area2, distance, gravity_constant=1.0):
    distance = max(float(distance), 0.001)
    return gravity_constant * np.power(area1 * area2, 0.6) / (distance ** 4)


def auto_adjust_parameters(df, target_edges_ratio=0.05):
    avg_area = df[AREA_COL].astype(float).mean()
    sample = df.head(min(10, len(df)))
    distances = []

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            distances.append(
                haversine_distance(
                    sample.iloc[i][LAT_COL], sample.iloc[i][LON_COL],
                    sample.iloc[j][LAT_COL], sample.iloc[j][LON_COL],
                )
            )

    avg_distance = np.mean(distances) if distances else 100.0
    typical_weight = np.power(avg_area ** 2, 0.6) / (avg_distance ** 4)
    gravity_constant = 0.0001 / typical_weight
    threshold = typical_weight * gravity_constant * target_edges_ratio
    return threshold, gravity_constant


def build_spatial_graph(df, threshold=None, gravity_constant=None):
    if threshold is None or gravity_constant is None:
        threshold, gravity_constant = auto_adjust_parameters(df)

    graph = nx.Graph()
    node_ids = df[ID_COL].tolist()

    for _, row in df.iterrows():
        attrs = {
            "area": float(row[AREA_COL]),
            "latitude": float(row[LAT_COL]),
            "longitude": float(row[LON_COL]),
            "label": row[LABEL_COL],
        }
        attrs.update({col: float(row[col]) for col in FEATURE_COLUMNS})
        graph.add_node(row[ID_COL], **attrs)

    for i in range(len(df)):
        row_i = df.iloc[i]
        for j in range(i + 1, len(df)):
            row_j = df.iloc[j]
            distance = haversine_distance(
                row_i[LAT_COL], row_i[LON_COL], row_j[LAT_COL], row_j[LON_COL]
            )
            weight = gravitational_weight(
                row_i[AREA_COL], row_j[AREA_COL], distance, gravity_constant
            )
            if weight > threshold:
                graph.add_edge(row_i[ID_COL], row_j[ID_COL], weight=weight, distance=distance)

    return graph, node_ids, threshold, gravity_constant


def get_lightgbm_leaf_embedding(x, y, n_estimators=100, num_leaves=64):
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        random_state=0,
        verbosity=-1,
    )
    model.fit(x, y)
    return model.predict(x, pred_leaf=True)


def compute_similarity_matrix(leaf_embedding):
    n_samples, n_trees = leaf_embedding.shape
    similarity = np.zeros((n_samples, n_samples), dtype=np.float32)
    for tree_idx in range(n_trees):
        leaf = leaf_embedding[:, tree_idx]
        similarity += leaf[:, None] == leaf[None, :]
    return similarity / n_trees


def build_similarity_adjacency(similarity, labels, k=8, positive_multiplier=100):
    n_samples = similarity.shape[0]
    adjacency = np.zeros((n_samples, n_samples), dtype=np.int8)

    for i in range(n_samples):
        n_neighbors = int(k * positive_multiplier) if labels[i] == 1 else k
        neighbors = np.argsort(similarity[i])[::-1][1:n_neighbors + 1]
        adjacency[i, neighbors] = 1

    return np.maximum(adjacency, adjacency.T)


def build_similarity_graph(df, labels, k=8, positive_multiplier=100):
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    leaf_embedding = get_lightgbm_leaf_embedding(x, labels)
    similarity = compute_similarity_matrix(leaf_embedding)
    adjacency = build_similarity_adjacency(similarity, labels, k, positive_multiplier)

    graph = nx.Graph()
    node_ids = df[ID_COL].tolist()
    graph.add_nodes_from(node_ids)

    rows, cols = np.where(np.triu(adjacency, k=1) == 1)
    graph.add_edges_from((node_ids[i], node_ids[j]) for i, j in zip(rows, cols))
    return graph


def graph_to_edge_index(graph, node_ids):
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    if not edges:
        raise ValueError("The graph contains no valid edges.")
    return torch.tensor(np.asarray(edges).T, dtype=torch.long)


def prepare_data(file_path, sheet_name=0):
    df = load_input_table(file_path, sheet_name)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[LABEL_COL].to_numpy())
    if len(label_encoder.classes_) != 2:
        raise ValueError("This implementation expects a binary class label.")

    scaler = StandardScaler()
    features = scaler.fit_transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    spatial_graph, node_ids, threshold, gravity_constant = build_spatial_graph(df)
    similarity_graph = build_similarity_graph(df, labels)

    data_a = Data(x=x.clone(), edge_index=graph_to_edge_index(spatial_graph, node_ids), y=y.clone())
    data_b = Data(x=x.clone(), edge_index=graph_to_edge_index(similarity_graph, node_ids), y=y.clone())

    data_a.node_ids = node_ids
    data_b.node_ids = node_ids
    data_a.label_classes = list(label_encoder.classes_)
    data_b.label_classes = list(label_encoder.classes_)

    info = {
        "n_nodes": len(node_ids),
        "n_features": len(FEATURE_COLUMNS),
        "spatial_edges": spatial_graph.number_of_edges(),
        "similarity_edges": similarity_graph.number_of_edges(),
        "spatial_threshold": threshold,
        "gravity_constant": gravity_constant,
    }
    return data_a, data_b, df, info


def add_isolated_self_loops(edge_index, num_nodes):
    all_nodes = torch.arange(num_nodes, device=edge_index.device)
    connected = edge_index.unique()
    isolated = all_nodes[~torch.isin(all_nodes, connected)]
    if isolated.numel() > 0:
        loops = isolated.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loops], dim=1)
    return edge_index


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, 32)
        self.conv2 = SAGEConv(32, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = add_isolated_self_loops(edge_index, x.size(0))
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GATEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()
        self.gat1 = GATConv(input_dim, 32, heads=num_heads)
        self.gat2 = GATConv(32 * num_heads, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = add_isolated_self_loops(edge_index, x.size(0))
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.gat2(x, edge_index)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        return self.fc3(x)


class MultiViewGNN(nn.Module):
    def __init__(self, input_dim=17, embedding_dim=16, num_classes=2):
        super().__init__()
        self.spatial_encoder = GraphSAGEEncoder(input_dim, embedding_dim)
        self.similarity_encoder = GATEncoder(input_dim, embedding_dim)
        self.classifier = MLP(embedding_dim, 32, num_classes)

    def forward(self, data_a, data_b):
        x = self.spatial_encoder(data_a) + self.similarity_encoder(data_b)
        return F.log_softmax(self.classifier(x), dim=1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def main():
    data_file = Path("./data")
    data_a, data_b, _, info = prepare_data(data_file)
    print("Data preparation complete")
    print(f"Nodes: {info['n_nodes']}")
    print(f"Features: {info['n_features']}")
    print(f"Spatial edges: {info['spatial_edges']}")
    print(f"Similarity edges: {info['similarity_edges']}")
    print(f"Spatial PyG data: {data_a}")
    print(f"Similarity PyG data: {data_b}")


if __name__ == "__main__":
    main()
