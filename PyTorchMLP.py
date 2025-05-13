import torch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units='a'):
        super(MultiLayerPerceptron, self).__init__()
        
        # Determine hidden layer size
        if hidden_units == 'a':
            hidden_dim = (input_dim + output_dim) // 2
        else:
            hidden_dim = int(hidden_units)

        # Define the layers
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

def load_data(filepath, device):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return (
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.int64).to(device),
        le
    )

def train_model(model, X_train, y_train, learning_rate=0.3, momentum=0.2, epochs=500):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

def perform_cross_validation(filepath, n_splits=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, label_encoder = load_data(filepath, device)
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    accuracies = []
    all_true, all_pred = [], []

    total_start_time = time.time()

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"Training Fold {fold}")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Initialize and train the model
        model = MultiLayerPerceptron(X_train.shape[1], len(torch.unique(y))).to(device)
        start_time = time.time()
        model = train_model(model, X_train, y_train)
        end_time = time.time()

        # Validate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_val.cpu(), predicted.cpu())
            accuracies.append(accuracy)
            all_true.extend(y_val.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        print(f"Training Time: {end_time - start_time:.4f} seconds")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Final report
    print("Average Accuracy: {:.4f}".format(np.mean(accuracies)))
    print("Standard Deviation: {:.4f}".format(np.std(accuracies)))
    print(f"Total Training Time: {total_duration:.4f} seconds")
    print("Classification Report:")
    print(classification_report(all_true, all_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_pred))

perform_cross_validation('hdcData.csv', n_splits=10)
