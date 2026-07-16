# LangGraph Chatbot using Groq LLM

A simple conversational AI chatbot built using **LangGraph**, **LangChain**, and **Groq LLM**. This project demonstrates how to build a stateful chatbot using LangGraph's graph-based workflow with in-memory checkpointing.

---

## Features

- LangGraph `StateGraph` implementation
- Stateful conversation using `MemorySaver`
- Integration with Groq LLM
- Environment variable configuration using `.env`
- Clean and modular code structure
- Interactive command-line chatbot
- Easy to extend with tools, agents, and RAG pipelines

---

## Project Structure

```text
open-source-contribution/
│
├── langgraph_demo.py      # Main chatbot implementation
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
├── .gitignore
└── README.md
```

---

## Architecture

```text
              START
                 │
                 ▼
        +----------------+
        |   Chatbot Node |
        +----------------+
                 │
                 ▼
                END
```

The chatbot consists of a single LangGraph node that:

1. Receives the conversation history.
2. Sends the messages to the Groq LLM.
3. Returns the generated response.
4. Stores the updated conversation using `MemorySaver`.

---

## Technologies Used

- Python 3.10+
- LangGraph
- LangChain
- Groq API
- python-dotenv

---

## Prerequisites

- Python 3.10 or above
- Groq API Key

Create a free API key from:

https://console.groq.com/keys

---

## Installation

### Clone the repository

```bash
git clone <repository-url>
cd open-source-contribution
```

### Create a virtual environment

#### Windows

```bash
python -m venv lang_env
```

Activate it

```bash
lang_env\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv lang_env
source lang_env/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key

MODEL_NAME=openai/gpt-oss-120b
```

> **Note:** Do not commit your `.env` file to GitHub.

---

## Running the Chatbot

Execute:

```bash
python langgraph_demo.py
```

---

## Sample Output

```text
============================================================
 LangGraph + Groq Chatbot
 Type 'exit' to quit
============================================================

You : Hello

Bot : Hello! How can I assist you today?

You : What is LangGraph?

Bot : LangGraph is an orchestration framework built on top of LangChain that enables developers to create stateful AI workflows using graph-based execution...
```

---

## Code Walkthrough

### Load Environment Variables

```python
load_dotenv()
```

Loads the Groq API key and model name from the `.env` file.

---

### Initialize the LLM

```python
llm = ChatGroq(
    model=os.getenv("MODEL_NAME"),
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)
```

Creates a Groq chat model.

---

### Define the State

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], "Conversation History"]
```

The graph state stores the complete conversation history.

---

### Chatbot Node

```python
def chatbot(state: ChatState):
```

This node:

- Receives the conversation history.
- Invokes the Groq LLM.
- Returns the updated conversation.

---

### Build the Graph

```python
graph_builder = StateGraph(ChatState)
```

The graph consists of:

```
START
   │
Chatbot
   │
 END
```

---

### Memory

```python
memory = MemorySaver()
```

Stores conversation checkpoints for each session.

---

### Execute the Graph

```python
result = graph.invoke(...)
```

Runs the graph and returns the chatbot response.

---

## Concepts Demonstrated

- LangGraph StateGraph
- START and END Nodes
- Typed State
- Memory Checkpointing
- Conversation History
- Groq Chat Models
- Environment Variables
- Interactive CLI Application

---

## Future Enhancements

- Multi-node LangGraph workflow
- Tool calling
- Web search integration
- Retrieval-Augmented Generation (RAG)
- Vector database integration
- Streaming responses
- Persistent memory using SQLite or PostgreSQL
- FastAPI deployment
- Docker support

---

## Requirements

```text
langgraph>=0.6.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-groq>=0.3.0
python-dotenv>=1.1.0
ipykernel>=6.30.0
jupyter>=1.1.0
```

---

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- LangChain Documentation: https://python.langchain.com/docs/
- Groq Documentation: https://console.groq.com/docs

---

## License

This project is licensed under the MIT License.

---

## Author

Developed as a demonstration project for learning **LangGraph**, **LangChain**, and **Groq LLM** integration.