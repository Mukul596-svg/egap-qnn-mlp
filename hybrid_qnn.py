import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Define quantum device
n_qubits = 4  # Adjust based on your needs
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode the input data
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum layers
    for layer in range(2):  # 2-layer quantum circuit
        # Entangling layer
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        
        # Rotation layer
        for i in range(n_qubits):
            qml.RY(weights[layer][i], wires=i)
    
    # Measure expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        weight_shapes = {"weights": (2, n_qubits)}  # 2 layers, n_qubits rotations each
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        x_reshaped = x.reshape(-1, self.n_qubits)
        return self.qlayer(x_reshaped)

class HybridNet(nn.Module):
    def __init__(self, input_dim, n_qubits=4):
        super().__init__()
        
        # Classical layers
        self.classical_pre = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_qubits)  # Reduce dimensions for quantum layer
        )
        
        # Quantum layer
        self.quantum = QuantumLayer(n_qubits)
        
        # Post-quantum classical layers
        self.classical_post = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.classical_pre(x)
        x = self.quantum(x)
        x = self.classical_post(x)
        return x

# Modified training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=300):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss/len(val_loader)
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# 1. Load and preprocess data
def load_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Drop cluster column if it exists
    if 'cluster' in train_df.columns:
        train_df = train_df.drop('cluster', axis=1)
        val_df = val_df.drop('cluster', axis=1)
        test_df = test_df.drop('cluster', axis=1)
    
    # Separate features and target
    X_train = train_df.drop('egap', axis=1)
    y_train = train_df['egap'].astype(np.float32)
    X_val = val_df.drop('egap', axis=1)
    y_val = val_df['egap'].astype(np.float32)
    X_test = test_df.drop('egap', axis=1)
    y_test = test_df['egap'].astype(np.float32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 2. Custom Dataset class
class MaterialsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
     # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data('train.csv', 'val.csv', 'test.csv')
    
    # Preprocessing
    numeric_features = [col for col in X_train.columns if col != 'spacegroup_relax']
    
    # Get all unique spacegroup values across all sets
    all_spacegroups = np.union1d(
        np.union1d(
            X_train['spacegroup_relax'].unique(),
            X_val['spacegroup_relax'].unique()
        ),
        X_test['spacegroup_relax'].unique()
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[all_spacegroups]), ['spacegroup_relax'])
        ])
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create data loaders
    train_dataset = MaterialsDataset(X_train_processed, y_train)
    val_dataset = MaterialsDataset(X_val_processed, y_val)
    test_dataset = MaterialsDataset(X_test_processed, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize hybrid model instead of MLP
    input_dim = X_train_processed.shape[1]
    model = HybridNet(input_dim).to(device)
    
    # Use slightly lower learning rate for quantum model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Evaluate the model
    print("\nTest Set Metrics:")
    metrics = evaluate_model(model, test_loader, device)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get predictions for plotting
    model.eval()
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze().cpu()
            all_predictions.extend(outputs.numpy())
            all_actuals.extend(batch_y.numpy())
    
    # Create and save the prediction plot
    plot_predictions(np.array(all_actuals), np.array(all_predictions), 'egap_predictions.png')
    print("\nPrediction plot saved as 'egap_predictions.png'")

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze().cpu()
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_predictions(actuals, predictions, save_path='prediction_plot.png'):
    """
    Create a scatter plot of predicted vs actual values with additional statistics
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(actuals, predictions, alpha=0.5)
    
    # Add diagonal line for perfect predictions
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    
    # Add R² value to plot
    r2 = r2_score(actuals, predictions)
    plt.text(0.1, 0.9, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()