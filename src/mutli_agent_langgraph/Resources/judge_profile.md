# QEA Copilot – Judge Profile (Project Context)

## Project Goals
- Multi-agent QEA Copilot that generates context-aware **manual test cases** and **automation test scripts** using LangGraph + RAG.
- Uses UI flow JSONs, user stories, business rules, and schemas from vector DBs.

## Output Requirements
### Test Case (Markdown or JSON)
- Sections: Title, Preconditions, Steps (numbered), Expected Results per step, Negative cases.
- Must be consistent with **Automation Test Store** (or target app).
- Avoid hallucinations; only use facts from retrieved contexts.

### Automation Script (Python + Playwright/Selenium)
- Deterministic, runnable; meaningful locators only; no TODOs.
- Include comments mapping steps → code.
- No credentials in code.

## JSON Schemas (if JSON mode)
- Test Case JSON must include: id, title, preconditions[], steps[], expected_results[], tags[].
- Script JSON (if used) must include: framework, language, files[], each with path, code.

## Domain Rules
- Use **page flow order** (before_page → after_page).
- Prefer **exact labels** from schema.
- Include **validation/edge** steps where applicable.

## Scoring Weights (total 10)
- Correctness 0.30, Completeness 0.25, Step Coverage 0.20, Formatting 0.10, Reasoning/Validation 0.10, Safety/Harm None 0.05 penalty.

## Hard Fails
- Fabricated endpoints/elements.
- Inconsistent flow order.
- JSON invalid when JSON is requested.
