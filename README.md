# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) assistant designed for medical question answering. This project leverages large language models and a document retrieval system to provide accurate, context-aware responses to medical queries.


## Getting Started

### Prerequisites

- uv
- conda

### Installation

```bash
git clone https://github.com/yourusername/medical-rag-chatbot.git
cd medical-rag-chatbot
uv sync
```

### Usage

```bash
python main.py
```

Follow the prompts to interact with the chatbot.

## Project Structure

```
medical-rag-chatbot/
├── data/                # Guideline pdf, abbreviation csv, fallvignetten, manually extracted table texts, whitespace processed guideline pickle, flowchart screenshots, page types json
├── results/             # Any results produced saved here
├── config/
│   └── config.yaml      # Configuration file
├── src/
│   ├── core/            # Core functionalities, inner sphere
│   ├── domain/          # Object definitions
│   ├── eval/            # Evaluation related methods
│   ├── experiments/     # All scripts including evaluations, some of which might not be up to date. Run by uv run python generation_retrieval_evaluation.py (use this script for evaluation)
│   ├── parsing/         # Pydantic models and parsing methods
│   ├── prompts/         # Prompt strings
│   ├── serve/           # Backend endpoints and VLM/LLM. For model endpoints: run inside GPU instance i.e. python qwen.py. For backend: uv run uvicorn app:app --port 7001 --reload --host 0.0.0.0
│   ├── services/        # Outer sphere. Retrieval service , question answering and ConversationService (this is implemented to align with frontend)
│   └── settings/        # Settings, requires .env and config.yaml
├── exllama.yml    # Exllama-Llama3.1 conda environment
├── vlm.yml    # QWEN conda environment
└── README.md           # Project documentation
```


## License

This project is licensed under the MIT License.