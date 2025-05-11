from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name, temperature=1, max_tokens=1000)

class State(TypedDict):
    messages: Annotated[list, add_messages] #the add message function will append the LLM's response messages to whatever messages are already in the state

graph_builder = StateGraph(State)

def chatbot(state:State):          #node takes current state as input
    return{"messages": [model.invoke(state["messages"])]}   #returns a dict containing an updates messages list under key "messages"

#Adding an entry point to tell the graph where to start its work each time it runs
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
#compile the graph
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][0].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

