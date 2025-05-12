from pprint import pprint

from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic

##checkpoints are better than chat memory
#good for error recovery, human in the loop workflow and time travel interactions

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name, temperature=1, max_tokens=1000)
memory = MemorySaver() ##in production you would use SqliteSave or PostgressSave and connect a db

#define a graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state:State):          #node takes current state as input
    return{"messages": [model.invoke(state["messages"])]}   #returns a dict containing an updates messages list under key "messages"

graph_builder = StateGraph(State)

#Adding an entry point to tell the graph where to start its work each time it runs
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

#compile
graph = graph_builder.compile(checkpointer=memory)

#pick a thread to use for chatbot
config = {
    "configurable":{
        "thread_id":"1"
    }
}

#call chatbot
user_input = "Hi there! My name is kite"

events = graph.stream(
    {"messages": [{"role":"user", "content": user_input}]}, config, stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

user_input = "Remember my name?"

events = graph.stream(
    {"messages": [{"role":"user", "content": user_input}]},  {"configurable": {"thread_id": "2"}}, stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
print(snapshot)
print(snapshot.next) ##since the graph ended this turn, `next` is empty. If you fetch a state from within a graph invocation, next tells which node will execute next)
