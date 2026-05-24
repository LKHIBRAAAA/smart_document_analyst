# Smart Document Analyst Architecture

## System Overview

The system is a multi-agent AI workflow for document analysis.

It includes:
- 1 orchestrator agent
- 3 specialist agents
- 4 tools
- 1 human-in-the-loop approval checkpoint

## Agents

### 1. OrchestratorAgent
Role:
Controls the workflow and coordinates all specialist agents.

Responsibilities:
- receives the input document text
- sends the text to the classifier agent
- sends the classification result to the extraction agent
- sends extracted information to the summary agent
- requests human approval before final acceptance
- stores workflow state

### 2. ClassifierAgent
Role:
Classifies the document into one of the supported categories.

Tool used:
- DocumentClassifierTool

Input:
- raw document text

Output:
- category
- confidence
- probabilities

### 3. ExtractionAgent
Role:
Extracts important structured information from the document.

Tool used:
- ExtractionTool

Input:
- raw document text
- predicted category

Output:
- extracted_fields
- dates
- amounts
- emails
- keywords

### 4. SummaryAgent
Role:
Generates a short summary of the document.

Tool used:
- OllamaSummaryTool
- SummaryTool (fallback)

Input:
- raw document text
- predicted category
- extracted information

Output:
- summary
- word_count
- key_points

Behavior:
- tries Ollama first using the local `phi3:latest` model
- falls back to the rule-based summary tool if Ollama is unavailable or returns an error

## Tools

### 1. DocumentClassifierTool
Purpose:
Uses the trained PyTorch model to classify document text.

Input schema:

```json
{
  "text": "string"
}
```

Output schema:

```json
{
  "category": "cv | invoice | report",
  "confidence": 0.0,
  "probabilities": {
    "cv": 0.0,
    "invoice": 0.0,
    "report": 0.0
  }
}
```

### 2. ExtractionTool
Purpose:
Extracts structured information depending on the document type.

Input schema:

```json
{
  "text": "string",
  "category": "cv | invoice | report"
}
```

Output schema:

```json
{
  "category": "string",
  "extracted_fields": {},
  "dates": [],
  "amounts": [],
  "emails": [],
  "keywords": []
}
```

### 3. OllamaSummaryTool
Purpose:
Generates a natural-language summary using the local Ollama API.

Input schema:

```json
{
  "text": "string",
  "category": "cv | invoice | report",
  "extracted_info": {}
}
```

Output schema:

```json
{
  "category": "string",
  "summary": "string",
  "word_count": 0,
  "key_points": []
}
```

### 4. SummaryTool
Purpose:
Generates a readable rule-based fallback summary from the document and extracted information.

Input schema:

```json
{
  "text": "string",
  "category": "cv | invoice | report",
  "extracted_info": {}
}
```

Output schema:

```json
{
  "category": "string",
  "summary": "string",
  "word_count": 0,
  "key_points": []
}
```

## Workflow

1. User provides document text
2. OrchestratorAgent starts the workflow
3. ClassifierAgent predicts the document category
4. ExtractionAgent extracts structured information
5. SummaryAgent generates a summary using Ollama
6. If Ollama fails, the system falls back to the rule-based summary tool
7. Human approval is requested
8. Final result is accepted or rejected

## Human-in-the-Loop Checkpoint

The system includes one mandatory checkpoint:
- the user must approve the analysis before the workflow is finalized

## Error Handling

The system handles:
- empty input text
- missing model files
- invalid category-dependent processing
- runtime failures during classification, extraction, and summarization
- Ollama API failures by switching to the fallback summary tool

## Logging

Every important agent action is logged in JSON format with timestamps in:

```text
logs/agent_logs.json
```
