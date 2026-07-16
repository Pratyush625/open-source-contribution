"""
Simple LangGraph Chatbot using Groq

Features
--------
✔ StateGraph
✔ Memory Checkpointer
✔ Conversation History
✔ Typed State
✔ START -> Chatbot -> END
"""

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------
# Initialize Groq LLM
# ---------------------------------------------------------

llm = ChatGroq(
    model=os.getenv("MODEL_NAME"),
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ---------------------------------------------------------
# Define Graph State
# ---------------------------------------------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], "Conversation History"]


# ---------------------------------------------------------
# Chatbot Node
# ---------------------------------------------------------


def chatbot(state: ChatState):

    response = llm.invoke(state["messages"])

    return {
        "messages": state["messages"] + [response]
    }


# ---------------------------------------------------------
# Build Graph
# ---------------------------------------------------------

graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

# ---------------------------------------------------------
# Memory
# ---------------------------------------------------------

memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)

# ---------------------------------------------------------
# Chat Session
# ---------------------------------------------------------

config = {
    "configurable": {
        "thread_id": "user_001"
    }
}

print("=" * 60)
print(" LangGraph + Groq Chatbot")
print(" Type 'exit' to quit")
print("=" * 60)

conversation = []

while True:

    user_input = input("\nYou : ")

    if user_input.lower() in ["exit", "quit"]:
        print("\nGoodbye 👋")
        break

    conversation.append(
        HumanMessage(content=user_input)
    )

    result = graph.invoke(
        {"messages": conversation},
        config=config,
    )

    assistant_message = result["messages"][-1]

    conversation.append(
        AIMessage(content=assistant_message.content)
    )

    print("\nBot :", assistant_message.content)