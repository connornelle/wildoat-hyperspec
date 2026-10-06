import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
TORCH_AVAILABLE = True

import umap
UMAP_AVAILABLE = True

#CNN architecture

class SpectralCNN(nn.Module):
    """
    1D CNN for hyperspectral regression.

    Architecture:
    Input (N, 1, n_bands)
    Conv blocks
    pooling
    bottleneck flatten
    scalar prediction
    """

    def __init__(self, n_bands: int = 300, bottleneck_dim: int = 16, dropout: float = 0.2):
        super().__init__()

        self.conv_block = nn.Sequential(
            #Block 1: broad patterns
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            #Block 2: fine patterns
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  #32, 1
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        x = self.conv_block(x)
        feature = self.bottleneck(x)
        out = self.head(feature)
        return out.squeeze(-1)

    def get_bottleneck_features(self, x):
        """Extract bottleneck features
        after conv block"""
        with torch.no_grad():
            x = self.conv_block(x)
            features = self.bottleneck(x)
        return features

#training

def train_cnn(X_train, y_train, X_val, y_val,
              bottleneck_dim=16, n_epochs=100, lr=1e-3,
              batch_size=32):

    device = torch.device("cpu")

    #fit scalar to train, apply to val
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = ((y_train - y_mean) /y_std).astype(np.float32)
    y_val_s = ((y_val- y_mean) / y_std).astype(np.float32)

    X_tr_t = torch.tensor(X_train_s[:, None, :]).to(device)
    y_tr_t = torch.tensor(y_train_s).to(device)
    X_va_t = torch.tensor(X_val_s[:, None, :]).to(device)
    y_va_t = torch.tensor(y_val_s).to(device)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training data dimension = {X_tr_t.shape}")

    model = SpectralCNN(n_bands=X_train.shape[1], bottleneck_dim=bottleneck_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}


    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = criterion(val_pred, y_va_t).item()

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1:3d} | train loss: {avg_train:.4f} | val loss: {val_loss:.4f}")

    return model, scaler, (y_mean, y_std), history, device

#bottleneck features

def extract_bottleneck(model, X, scaler, device):
    X_s = scaler.transform(X).astype(np.float32)
    X_t= torch.tensor(X_s[:, None, :]).to(device)
    model.eval()
    features = model.get_bottleneck_features(X_t)
    return features.cpu().numpy()

#umap

def run_umap(features, n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(features)

#model metrics evaluation

def evaluate_predictions(model, X_val, y_val, scaler, y_stats, device):
    y_mean, y_std = y_stats
    X_s =scaler.transform(X_val).astype(np.float32)
    X_t = torch.tensor(X_s[:, None, :]).to(device)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_t).cpu().numpy()
    pred = pred_s * y_std + y_mean
    r2 = r2_score(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return pred, r2, rmse

