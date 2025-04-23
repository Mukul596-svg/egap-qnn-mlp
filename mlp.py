import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. Load and preprocess data
def load_data(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Drop cluster column if it exists
    if 'cluster' in train_df.columns:
        train_df = train_df.drop('cluster', axis=1)
    
    # Separate features and target
    X_train = train_df.drop('egap', axis=1)
    y_train = train_df['egap']
    X_val = val_df.drop('egap', axis=1)
    y_val = val_df['egap']
    X_test = test_df.drop('egap', axis=1)
    y_test = test_df['egap']
    
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

# 3. Neural Network Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

# 4. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=300):
    train_losses = []
    val_losses = []
    
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
        
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# 5. Evaluation function
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

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data('train.csv', 'val.csv', 'test.csv')
    
    # Preprocessing
    numeric_features = [col for col in X_train.columns if col != 'spacegroup_relax']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False), ['spacegroup_relax'])
        ])
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create data loaders
    train_dataset = MaterialsDataset(X_train_processed, y_train.values)
    val_dataset = MaterialsDataset(X_val_processed, y_val.values)
    test_dataset = MaterialsDataset(X_test_processed, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    input_dim = X_train_processed.shape[1]
    model = MLP(input_dim).to(device)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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

if __name__ == "__main__":
    main()