"""
Smart Document Analyst - Model Training Script
================================================
Trains a PyTorch neural network for document classification (CV, Invoice, Report)
Uses TF-IDF for feature extraction + Neural Network classifier
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class DocumentClassifierNN(nn.Module):
    """
    Neural Network for Document Classification
    Architecture: Input -> Dense -> ReLU -> Dropout -> Dense -> ReLU -> Dropout -> Output
    """
    def __init__(self, input_size, num_classes):
        super(DocumentClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_and_preprocess_data(data_path):
    """
    Load dataset and preprocess text
    """
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Labels: {df['label'].unique()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def create_tfidf_features(train_texts, test_texts):
    """
    Create TF-IDF features from text data
    """
    print("\nCreating TF-IDF features...")
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    
    # Fit on training data and transform both
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"TF-IDF features created: {X_train.shape[1]} features")
    
    return vectorizer, X_train, X_test


def train_model(X_train, y_train, X_val, y_val, input_size, num_classes):
    """
    Train the PyTorch neural network
    """
    print("\n" + "=" * 60)
    print("Training Neural Network...")
    print("=" * 60)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.toarray())
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val.toarray())
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = DocumentClassifierNN(input_size, num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)
        
        train_accuracy = train_correct / train_total
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model
    """
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)
    
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test.toarray())
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predicted.numpy())
    conf_matrix = confusion_matrix(y_test, predicted.numpy())
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"Labels: {list(label_encoder.classes_)}")
    print(conf_matrix)
    
    print(f"\nClassification Report:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    report = classification_report(
        y_test,
        predicted.numpy(),
        target_names=target_names,
        output_dict=True
    )
    print(classification_report(y_test, predicted.numpy(), target_names=target_names))

    # Save evaluation results for the final project submission
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluation_path = os.path.join(script_dir, "evaluation_results.json")

    evaluation_results = {
        "accuracy": float(accuracy),
        "classes": target_names,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report
    }

    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\nEvaluation results saved to: {evaluation_path}")

    return accuracy, conf_matrix


def save_model_artifacts(model, vectorizer, label_encoder, model_path):
    """
    Save model, vectorizer, and label encoder
    """
    print("\n" + "=" * 60)
    print("Saving Model Artifacts...")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "model")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save PyTorch model
    model_file = os.path.join(model_dir, "document_classifier.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.fc1.in_features,
        'num_classes': model.fc3.out_features
    }, model_file)
    print(f"Model saved to: {model_file}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, "document_classifier_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save label encoder
    label_encoder_path = os.path.join(model_dir, "document_classifier_label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {label_encoder_path}")
    
    return vectorizer_path, label_encoder_path


def main():
    """
    Main training function
    """
    print("\n" + "=" * 60)
    print("SMART DOCUMENT ANALYST - MODEL TRAINING")
    print("=" * 60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Paths
    data_path = os.path.join(project_dir, "data", "dataset.csv")
    model_path = os.path.join(project_dir, "model", "document_classifier.pt")
    
    # Load data
    df = load_and_preprocess_data(data_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['text'].apply(lambda x: df[df['text'] == x]['label'].iloc[0]))
    
    # Get labels properly
    y = label_encoder.fit_transform(df['label'])
    print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'].values, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train_text)} samples")
    print(f"Test set: {len(X_test_text)} samples")
    
    # Create TF-IDF features
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(X_train_text, X_test_text)
    
    # Train model
    model = train_model(
        X_train_tfidf, y_train,
        X_test_tfidf, y_test,
        input_size=X_train_tfidf.shape[1],
        num_classes=len(label_encoder.classes_)
    )
    
    # Evaluate model
    accuracy, conf_matrix = evaluate_model(model, X_test_tfidf, y_test, label_encoder)
    
    # Save artifacts
    save_model_artifacts(model, vectorizer, label_encoder, model_path)


if __name__ == "__main__":
    main()