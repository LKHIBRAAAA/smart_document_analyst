# LIVE DEMO LINK ===> https://youtu.be/DtmOitgr61c

# Smart Document Analyst

A multi-agent AI system for document classification and analysis using PyTorch deep learning.

## Project Overview

**Smart Document Analyst** is an Integrated Project for building Multi-Agent AI Systems. The system classifies documents (CV, Invoice, Report) using a PyTorch neural network with TF-IDF features, extracts key information, generates summaries with Ollama, and includes human-in-the-loop approval before final report generation.

See `ARCHITECTURE.md` for the agent roles, workflow, tool schemas, and system design.

## Features

- **Multi-Agent System**: 4 specialized agents working together
  - OrchestratorAgent: Coordinates workflow
  - ClassifierAgent: Classifies documents using PyTorch
  - ExtractionAgent: Extracts key information
  - SummaryAgent: Generates summaries
- **PyTorch Deep Learning Model**: Neural network classifier
- **TF-IDF + Neural Network**: Text feature extraction
- **Ollama LLM Summarization**: Local `phi3:latest` model used by the summary agent
- **Human-in-the-loop Checkpoint**: User approval before final report
- **JSON Logging**: Timestamped logs for all agent actions
- **Error Handling**: Empty input, missing files, invalid output
- **Simple CLI Application**: Easy to use interface

## Project Structure

```text
smart_document_analyst/
|-- app/
|   |-- agents.py
|   |-- logger.py
|   |-- main.py
|   `-- tools.py
|-- data/
|   `-- dataset.csv
|-- logs/
|   `-- agent_logs.json
|-- model/
|   |-- document_classifier.pt
|   |-- document_classifier_label_encoder.pkl
|   |-- document_classifier_vectorizer.pkl
|   `-- train_model.py
|-- ARCHITECTURE.md
|-- final_report.txt
|-- README.md
`-- requirements.txt
```

## Installation

1. Open a terminal in the project folder:

```bash
cd smart_document_analyst
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and the local summary model is available:

```bash
ollama list
```

The project is configured to use:

```text
phi3:latest
```

4. Train the model if the model files are missing or if you want to retrain it:

```bash
python model/train_model.py
```

5. Run the application:

```bash
python app/main.py
```

## Required Files

The project must contain the following files before running the application:

```text
data/dataset.csv
model/document_classifier.pt
model/document_classifier_vectorizer.pkl
model/document_classifier_label_encoder.pkl
```

If the model files are missing, run:

```bash
python model/train_model.py
```

## Usage

### Menu Options

1. **Analyze a document**: Enter document text for classification
2. **Train the model**: Retrain the PyTorch model
3. **View logs**: View agent activity logs
4. **Generate final report**: Create final analysis report
5. **Exit**: Exit the application

### Document Types

The system classifies documents into three categories:

- **CV**: Resumes and professional profiles
- **Invoice**: Billing and payment documents
- **Report**: Financial and business reports

## Model Architecture

```text
Input (TF-IDF features)
  |
Dense(256) + ReLU + Dropout(0.3)
  |
Dense(128) + ReLU + Dropout(0.3)
  |
Dense(num_classes) + Softmax
  |
Output: Classification
```

## Agent Workflow

```text
Document Text
  |
[ClassifierAgent] -> Classify (CV/Invoice/Report)
  |
[ExtractionAgent] -> Extract key information
  |
[SummaryAgent] -> Generate summary with Ollama (fallback to rule-based summary if unavailable)
  |
[Human Approval] -> User confirms analysis
  |
[Final Report] -> Generate report
```

## Error Handling

The system handles:
- Empty document input
- Missing model files
- Invalid classification output
- File I/O errors

## Logging

All agent actions are logged in JSON format with timestamps:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "agent": "ClassifierAgent",
  "action": "classifying_document",
  "details": {"text_length": 150}
}
```

## Example Usage

```text
============================================================
       SMART DOCUMENT ANALYST
       Multi-Agent AI System
============================================================

MAIN MENU
----------------------------------------
1. Analyze a document
2. Train the model
3. View logs
4. Generate final report
5. Exit
----------------------------------------

Enter your choice (1-5): 1

Enter the document text (paste or type):
----------------------------------------
(Paste your document here)
(Press Enter twice to finish)
```

## Technical Details

- **Framework**: PyTorch
- **Feature Extraction**: TF-IDF (1000 features, unigrams + bigrams)
- **Training**: Adam optimizer, CrossEntropyLoss
- **Early Stopping**: Patience of 10 epochs
- **Evaluation**: Accuracy, Confusion Matrix
- **LLM Backend**: Ollama using `phi3:latest` for local summary generation

## Evaluation Results

The trained PyTorch model is evaluated automatically after training.

The evaluation artifact is saved in:

```text
model/evaluation_results.json
```

This file contains:
- accuracy
- class labels
- confusion matrix
- classification report

## Requirements Met

- Multi-agent system (4 agents)
- Orchestrator agent
- At least 2 specialist agents
- PyTorch deep learning model
- DL model as functional tool
- Ollama LLM integrated into the summary agent
- Human-in-the-loop checkpoint
- JSON logging with timestamps
- Error handling
- Final report generation
- Simple CLI application
- Model evaluation saved to JSON (accuracy, confusion matrix, classification report)

## License

This project is for educational purposes.

## Author

Created for: Integrated Project - Building Multi-Agent AI Systems
