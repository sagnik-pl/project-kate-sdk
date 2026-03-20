# Getting Started with KATE SDK

This guide covers how to connect any AI agent to KATE for automatic evaluation, tracing, and monitoring.

---

## Overview

KATE SDK provides three capabilities:

1. **Tracing** — Capture every LLM call with `@kate_sdk.trace`
2. **Run lifecycle** — Group traces into runs with `kate_sdk.run()`
3. **Auto-eval** — Automatically evaluate each run when it completes

## Install

```bash
pip install kate-sdk
```

For auto-instrumentation of specific providers:

```bash
pip install kate-sdk[openai]                # OpenAI SDK
pip install kate-sdk[anthropic-instrument]  # Anthropic SDK
pip install kate-sdk[langchain]             # LangChain / LangGraph
pip install kate-sdk[all]                   # All providers
```

---

## Quick Start

### 1. Initialize the SDK

```python
import kate_sdk

kate_sdk.init(
    api_url="http://localhost:8000",   # KATE server URL
    api_key="your-api-key",            # Your KATE API key
    agent_id="your-agent-uuid",        # Your agent's ID in KATE
)
```

Or set environment variables (`KATE_API_URL`, `KATE_API_KEY`, `KATE_AGENT_ID`) and call `kate_sdk.init()` with no arguments.

### 2. Trace LLM calls

Use the `@kate_sdk.trace` decorator on any function that calls an LLM:

```python
@kate_sdk.trace("summarize")
def summarize(text: str) -> str:
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    ).content[0].text
```

### 3. Run your agent

Wrap your agent execution in a `kate_sdk.run()` context:

```python
async with kate_sdk.run() as ctx:
    result = summarize("Today's top news stories...")
    ctx.output(result)
```

On exit, the SDK:
- Marks the run complete
- Triggers auto-evaluation
- Scores each traced LLM call

---

## Auto-Instrumentation

For frameworks with OTel support, enable auto-instrumentation to capture all LLM calls without decorators:

```python
kate_sdk.init(auto_instrument=True)
```

This works with LangChain, OpenAI SDK, Anthropic SDK, and other supported providers (requires the corresponding `[extra]` installed).

---

## Local Eval (no server)

Run evaluations locally without a KATE server:

```python
from kate_sdk.local import LocalRunner

runner = LocalRunner(agent_fn=my_agent)
results = await runner.run(test_cases=[
    {"input": "Summarize the news", "expected": "A concise summary..."},
])
runner.print_results(results)
```

---

## What KATE Evaluates

KATE classifies each traced LLM node and selects appropriate metrics:

| Classification | Metrics | Example |
|---|---|---|
| **SUMMARIZATION** | Faithfulness, Summarization | Summary generators |
| **RAG_QA** | Faithfulness, Answer Relevancy | Q&A with retrieval |
| **SELECTION_RANKING** | GEval (auto-generated) | Ranking, filtering |
| **EXTRACTION** | GEval (auto-generated) | Data extraction |
| **TRANSFORMATION** | None (no LLM judgment) | Format conversion |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KATE_API_URL` | `http://localhost:8000` | KATE server URL |
| `KATE_API_KEY` | — | Your API key |
| `KATE_AGENT_ID` | — | Your agent's UUID |

---

## Examples

See the [examples/](../examples/) directory for complete, runnable agents.
