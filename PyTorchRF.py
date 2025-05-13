import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time
import torch.nn as nn
import torch.optim as optim

# Random Forest-like Model (using Multiple Decision Trees)
class RandomForestModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_trees=100):
        super().__init__()
        self.num_trees = num_trees
        self.trees = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            ) for _ in range(num_trees)
        ])
    
    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        # Average predictions from all trees (bagging)
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output

def load_csv_data(filepath, device):
    # Load CSV and preprocess
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Encode categorical labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Convert to Torch tensors and move to device (CUDA if available)
    return torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device), le

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Lists to track training progress
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Verbose logging
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss.item():.4f}")
            print(f"Validation Loss: {val_loss.item():.4f}")
            print("-" * 30)
    
    return model, {
        'train_losses': train_losses, 
        'val_losses': val_losses
    }

def perform_cross_validation(filepath, n_splits=10, epochs=50):
    # Ensure CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data and move to appropriate device
    X, y, label_encoder = load_csv_data(filepath, device)
    
    # Prepare K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    # Store results
    fold_accuracies = []
    fold_training_histories = []
    all_true_labels = []
    all_predicted_labels = []
    
    # Time tracking for total training duration
    start_time = time.time()

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold} TRAINING")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Initialize model and move to device
        model = RandomForestModel(X_train.shape[1], len(torch.unique(y)), num_trees=100).to(device)
        
        # Train model with specified epochs
        model, training_history = train_model(
            model, 
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            epochs=epochs
        )
        fold_training_histories.append(training_history)
        
        # Validate
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs, 1)
            
            # Move predictions and true labels back to CPU for sklearn metrics
            predicted_cpu = predicted.cpu().numpy()
            y_val_cpu = y_val.cpu().numpy()
            
            accuracy = accuracy_score(y_val_cpu, predicted_cpu)
            fold_accuracies.append(accuracy)
            
            # Collect predictions and true labels for final report
            all_true_labels.extend(y_val_cpu)
            all_predicted_labels.extend(predicted_cpu)
            
            print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    
    # Total training time
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    
    # Overall results
    print(f"\nAverage Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")
    
    # Generate comprehensive classification report
    class_names = label_encoder.classes_
    print("\nDetailed Classification Report:")
    print(classification_report(
        all_true_labels, 
        all_predicted_labels, 
        target_names=class_names,
        zero_division=1
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'fold_accuracies': fold_accuracies,
        'training_histories': fold_training_histories
    }
results = perform_cross_validation('hdcData.csv', epochs=100)
