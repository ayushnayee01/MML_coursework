import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import find_file, load_data, load_hidden_X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_blocks: int = 4):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)

def train_resnet(X_train, y_train, X_val=None, y_val=None, epochs=85, lr=1e-3, batch_size=256):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TabularResNet(in_dim=X_train.shape[1], hidden_dim=256, num_blocks=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if X_val is not None and y_val is not None:
            model.eval()
            X_v = scaler.transform(X_val)
            with torch.no_grad():
                logits = model(torch.tensor(X_v, dtype=torch.float32).to(device))
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
            acc = accuracy_score(y_val, preds)
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_acc

def predict_resnet(model, scaler, X_test):
    model.eval()
    X_in = scaler.transform(X_test)
    with torch.no_grad():
        logits = model(torch.tensor(X_in, dtype=torch.float32).to(device))
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

def main():
    print("=== MODEL 6: PYTORCH TABULAR RESNET ===")
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs("hiddenlabels", exist_ok=True)
    output_files = {
        '20': 'y_predicted_20.npy',
        '30': 'y_predicted_30.npy',
        '40': 'y_predicted_40.npy'
    }

    for key in ['20', '30', '40']:
        X, y, X_train, X_val, y_train, y_val = load_data(key)
        _, _, acc = train_resnet(X_train, y_train, X_val, y_val, epochs=85, lr=1e-3)
        print(f"  Kryptonite-{key:8s} | Val Accuracy: {acc*100:.2f}%")

        # Retrain on 100% of labeled data and save hidden predictions
        final_net, final_scaler, _ = train_resnet(X, y, epochs=85, lr=1e-3)
        preds = predict_resnet(final_net, final_scaler, load_hidden_X(key))
        out_path = os.path.join("hiddenlabels", output_files[key])
        np.save(out_path, preds)

    check_script = find_file("check_submission.py")
    if check_script:
        subprocess.run([sys.executable, check_script], check=False)

if __name__ == "__main__":
    main()
