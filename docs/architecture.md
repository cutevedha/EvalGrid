# Architecture

## Core flow
1. Load test cases
2. Run deterministic checks
3. Run semantic checks
4. Run judge-model checks
5. Apply PII and prompt-injection guards
6. Emit reports and gate releases

## Production modules
- `evals/llm_judge.py`
- `evals/json_schema.py`
- `guards/pii.py`
- `guards/prompt_injection.py`
- `reports/html_report.py`
- `synthetic/redteam.py`
