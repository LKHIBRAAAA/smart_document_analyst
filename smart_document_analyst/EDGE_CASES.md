# Edge Case Testing

## Purpose

This document lists edge cases used to evaluate the robustness of the Smart Document Analyst system.

The goal is to show that the system does not only work on ideal inputs, but also handles weak, incomplete, or ambiguous inputs in a controlled way.

## Edge Cases Covered

### 1. Empty input

Input:

```text
""
```

Expected behavior:
- the system should reject the input
- no crash should occur
- a clear error message should be returned

### 2. Whitespace-only input

Input:

```text
"   "
```

Expected behavior:
- the system should reject the input
- no crash should occur

### 3. Very short input

Input:

```text
"Invoice"
```

Expected behavior:
- the system should still return a classification result
- low confidence is acceptable
- no crash should occur

### 4. Mixed or ambiguous content

Input:

```text
"John Smith submitted an invoice for consulting work in Q1 2024 with a total of 2500 MAD."
```

Expected behavior:
- the system should classify the text into one category
- extraction and summary should still complete
- output may be imperfect but should remain valid

### 5. Invoice without amount

Input:

```text
"Invoice #9001 for design services. Client: Nova. Contact: billing@nova.com"
```

Expected behavior:
- invoice fields should be partially extracted
- summary should still be generated
- missing amount should not crash the workflow

### 6. CV without name

Input:

```text
"Experienced Python developer with 4 years experience in machine learning and Docker."
```

Expected behavior:
- classification should still work
- summary should use fallback values if name is missing

### 7. Report without metrics

Input:

```text
"Quarterly business report discussing internal process improvements and planning priorities."
```

Expected behavior:
- the text should still be summarized
- missing revenue or growth should not crash the system

### 8. Unknown or off-domain text

Input:

```text
"The weather is pleasant today and the city center is crowded."
```

Expected behavior:
- the classifier will still choose one of the known categories
- low confidence is acceptable
- output should remain structurally valid

## Execution

Edge-case checks can be run with:

```bash
python tests/run_edge_case_checks.py
```

## Result Artifact

The execution script saves results to:

```text
logs/edge_case_results.json
```
