# AI Research & Report Agent

An intelligent multi-agent system built with **LangGraph** that routes user queries between an internal **HR Knowledge Base (RAG)** and **web research**, evaluates its own outputs, and iteratively improves them until quality criteria are met.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Knowledge Base](#knowledge-base)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**AI Research & Report Agent** is a stateful multi-agent system that:

1. **Routes** incoming user queries to the right handler (internal HR knowledge base or external web research).
2. **Retrieves** relevant documents from a Chroma vector store (RAG) for HR-related questions.
3. **Searches** the web via Tavily for general or up-to-date information.
4. **Generates** coherent answers with citations.
5. **Evaluates** its own output and iterates until the response meets quality standards.

The system is designed to answer **HR policy questions** for employees of *ArianaTech* (a fictional Iranian tech company used as the demo organization) and general knowledge questions that require web research.

---

## Features

- ✅ **Intelligent Query Routing** — classifies queries into `RAG`, `generate`, or `unknown`
- ✅ **RAG Pipeline** — Chroma vector store + HuggingFace embeddings (`all-mpnet-base-v2`)
- ✅ **Multi-Query Retrieval** — generates multiple search queries per user question
- ✅ **Vote-based Ranking** — ranks retrieved chunks by how many queries returned them
- ✅ **Web Search Integration** — Tavily search for real-time information
- ✅ **Generate–Evaluate Loop** — self-critique and refinement until output is approved
- ✅ **Structured Outputs** — Pydantic schemas for reliable routing and evaluation
- ✅ **Persistent Memory** — checkpointing with LangGraph (SQLite / in-memory)
- ✅ **Graph Visualization** — auto-generated Mermaid diagram of the workflow
- ✅ **Citation-aware Answers** — every HR answer cites its source file

---

## Architecture

```
                          ┌──────────────┐
                          │     START    │
                          └──────┬───────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │   Router (LLM)     │
                       │  RAG / generate /  │
                       │      unknown       │
                       └─────┬────┬─────┬───┘
                             │    │     │
              ┌──────────────┘    │     └──────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
       ┌────────────┐      ┌─────────────┐           END
       │    RAG     │      │  generate   │
       │  (Chroma + │      │  (Tavily +  │
       │   LLM)     │      │    LLM)     │
       └─────┬──────┘      └──────┬──────┘
             │                    │
             ▼                    ▼
            END            ┌────────────┐
                           │    eval    │
                           │ (critique) │
                           └──────┬─────┘
                                  │
                        ┌─────────┴─────────┐
                        │                   │
                   Ok_research         Change_research
                        │                   │
                        ▼                   ▼
                       END             back to generate
```

### Nodes

| Node | Purpose |
|------|---------|
| **router** | Classifies the user query into `RAG`, `generate`, or `unknown` |
| **RAG** | Retrieves HR documents from Chroma, ranks them, generates a cited answer |
| **generate** | Searches the web via Tavily, synthesizes a research passage |
| **eval** | Critiques the generated output; decides whether to iterate or finish |

---

## Tech Stack

| Category | Tool |
|----------|------|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **LLM Framework** | [LangChain](https://github.com/langchain-ai/langchain) |
| **Vector Store** | [Chroma](https://github.com/chroma-core/chroma) |
| **Embeddings** | [HuggingFace `sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| **Web Search** | [Tavily](https://tavily.com/) |
| **LLM Providers** | OpenRouter (Nemotron, GPT-4o-mini, Claude Haiku, etc.) |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) |
| **Env Management** | [uv](https://github.com/astral-sh/uv) |
| **Language** | Python 3.10+ |

---

## Project Structure

```
AI_Research_&_Report_Agent/
├── main.py                     # Entry point, graph builder, test runner
├── model.py                    # LLM factory (per-purpose model selection)
├── RAG.py                      # Loader, splitter, embeddings, retriever, ranker
├── graph.py                    # Node definitions, prompts, routing logic
├── .env                        # API keys (not committed)
├── .gitignore
├── pyproject.toml
├── uv.lock
├── chroma_db/                  # Persisted vector store
├── HR_Rules/                   # HR knowledge base (18 .txt files)
│   ├── 00_index.txt
│   ├── 01_company_overview.txt
│   ├── 02_leave_and_time_off.txt
│   ├── ...
│   └── 18_faq.txt
├── graph.png                   # Auto-generated workflow diagram
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.10+**
- [`uv`](https://github.com/astral-sh/uv) package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI_Research_&_Report_Agent.git
cd AI_Research_&_Report_Agent

# 2. Install dependencies
uv sync

# 3. Copy the example env file
cp .env.example .env
# Then edit .env and add your API keys

# 4. Run the agent
uv run main.py
```

### Key dependencies

```bash
uv add langgraph langchain langchain-openai langchain-community \
       langchain-chroma langchain-huggingface langchain-tavily \
       chromadb sentence-transformers python-dotenv pydantic \
       langgraph-checkpoint-sqlite
```

---

## Configuration

Create a `.env` file in the project root:

```env
# OpenRouter (or any OpenAI-compatible provider)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Tavily web search
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx

# Optional: HuggingFace token (faster model downloads)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> **Note:** HuggingFace models download on first run. Setting `HF_TOKEN` avoids rate-limit warnings and speeds up downloads.

---

## Usage

### Run a single query

```python
from langchain_core.messages import HumanMessage
from main import graph

config = {"configurable": {"thread_id": "user-123"}}

for chunk in graph.stream(
    {"messages": [HumanMessage(content="How many annual leave days do I get?")]},
    config=config,
    stream_mode="updates",
):
    print(chunk)
```

### Example queries

| Query | Route | Expected |
|-------|-------|----------|
| `How many annual leave days do I get?` | RAG | 26 working days, cited from `02_leave_and_time_off.txt` |
| `Can I work remotely?` | RAG | Hybrid model, 2 remote days/week |
| `What are the latest trends in HR tech?` | generate | Tavily search + synthesis |
| `hello` | unknown | END (no processing) |

### Visualize the graph

```python
graph.get_graph().draw_mermaid_png(output_file_path="./graph.png")
```

Or, if PNG rendering fails:

```python
print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live
```

---

## How It Works

### 1. Router

The router uses an LLM with structured output to classify the query:

```python
class RouterSchema(BaseModel):
    route: Literal["RAG", "generate", "unknown"]
    reason: str
```

Rules:
- **RAG** — questions about ArianaTech's internal HR policies
- **generate** — general knowledge, news, trends
- **unknown** — off-topic or unclear input

### 2. RAG Pipeline

```
User question
     │
     ▼
Query Generator (LLM)  ──►  1–3 refined search queries
     │
     ▼
Chroma Retriever (k=5 per query)
     │
     ▼
Vote-based Ranker  ──►  top-k chunks
     │
     ▼
Answer Generator (LLM + citations)
```

**Key functions:**

- `rag_retriever()` — loads or builds the Chroma vector store
- `rank_by_votes()` — ranks chunks by how many queries retrieved them
- `format_context()` — injects source labels into the context

### 3. Generate–Evaluate Loop

```
generate ──► eval ──► (Ok_research) ──► END
                 │
                 └──► (Change_research) ──► generate (retry)
```

- **generate** synthesizes a research passage from Tavily results
- **eval** critiques it against relevance, accuracy, completeness, clarity, and hallucination
- **Max 3 attempts** to prevent infinite loops
- Search results are cached in state so Tavily is only called once per question

---

## Knowledge Base

The `HR_Rules/` directory contains 18 plain-text files forming the HR knowledge base for **ArianaTech**, a fictional Iranian tech company:

| # | File | Topic |
|---|------|-------|
| 00 | `00_index.txt` | Table of contents |
| 01 | `01_company_overview.txt` | Company profile |
| 02 | `02_leave_and_time_off.txt` | Leave policies |
| 03 | `03_working_hours_and_attendance.txt` | Working hours |
| 04 | `04_compensation_and_benefits.txt` | Salary & benefits |
| 05 | `05_performance_and_career.txt` | Performance reviews |
| 06 | `06_onboarding_and_probation.txt` | Onboarding |
| 07 | `07_resignation_and_termination.txt` | Resignation |
| 08 | `08_code_of_conduct.txt` | Code of conduct |
| 09 | `09_health_safety_wellbeing.txt` | Health & safety |
| 10 | `10_remote_work_policy.txt` | Remote work |
| 11 | `11_it_and_equipment.txt` | IT & equipment |
| 12 | `12_travel_and_expenses.txt` | Travel policy |
| 13 | `13_diversity_and_inclusion.txt` | Diversity & inclusion |
| 14 | `14_training_and_development.txt` | Training |
| 15 | `15_employee_relations.txt` | Employee relations |
| 16 | `16_payroll_and_documentation.txt` | Payroll |
| 17 | `17_general_policies.txt` | General policies |
| 18 | `18_faq.txt` | FAQ |

**Chunking strategy:** `RecursiveCharacterTextSplitter` with `chunk_size=800` and `chunk_overlap=100`.

---

## Roadmap

- [x] RAG pipeline with Chroma
- [x] Multi-query retrieval
- [x] Vote-based ranking
- [x] Tavily web search integration
- [x] Generate–Evaluate loop
- [x] Router with structured output
- [x] Graph visualization
- [ ] Persistent memory via SQLite checkpointer
- [ ] Multi-hop retrieval (cross-document reasoning)
- [ ] Streaming responses to the UI
- [ ] Evaluation suite with benchmark Q&A pairs
- [ ] FastAPI backend + React frontend
- [ ] Persian language support in prompts

---

## Known Issues

| Issue | Cause | Workaround |
|-------|-------|-----------|
| `Invalid JSON` from `with_structured_output` | Free OpenRouter models don't reliably support structured outputs | Use `gpt-4o-mini` or `claude-haiku` for eval/router nodes |
| High `reasoning_tokens` in responses | Reasoning models over-think simple HR questions | Set `reasoning_effort="low"` or switch to non-reasoning models |
| Memory lost between runs | `MemorySaver` is in-process only | Use `SqliteSaver` for persistence |
| HF Hub rate-limit warning | No `HF_TOKEN` set | Add `HF_TOKEN` to `.env` |

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please make sure your code passes any existing tests and follows the existing style.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://github.com/langchain-ai/langgraph)
- [Chroma](https://github.com/chroma-core/chroma) for the vector store
- [Tavily](https://tavily.com/) for web search
- [OpenRouter](https://openrouter.ai/) for LLM access
- [HuggingFace](https://huggingface.co/) for embeddings

---

## Contact

**Author:** Amirali
**Project:** AI Research & Report Agent
**Repository:** [github.com/<your-username>/AI_Research_&_Report_Agent](https://github.com/<your-username>/AI_Research_&_Report_Agent)

---

⭐ If you find this project useful, please consider giving it a star!