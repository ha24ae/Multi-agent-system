#websearch
from langchain_tavily import TavilySearch

#previous
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic

#tools
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"
model = ChatAnthropic(api_key =anthropic_api_key, model=llm_name, temperature=1, max_tokens=1000)
memory = MemorySaver()

#define state graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

#similar to pythons input() function , calling 'interrupt' inside the tool will pause execution.
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool = TavilySearch(max_result=2)
tools= [tool, human_assistance]
model_with_tools= model.bind_tools(tools)

def chatbot(state:State): #node takes current state as input
    return {"messages": [model_with_tools.invoke(state["messages"])]}  #returns a dict containing an updates messages list under key "messages"

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START, "chatbot")

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=memory)


####testing chatbot
config = {"configurable": {"thread_id": "1"}}

user_input = "Hi, I require some expert guidance in building an AI agent?"

#as the chatbot is interrupted during use in human_assistance tool
##we need to resume execution
##to resume execution we need to pass a Command object containing data expected by the toll.
#the format of the data can be customised based on needs .
##for this example dictionary with key "data"

human_response = ("We are here to help! Check out LangGraphs website")

human_command = Command(resume={"data": human_response})


events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        #print(event)
        event["messages"][-1].pretty_print()

#chatbot generated a toll call, but execution was interrupted
#inspect graph state
snapshot = graph.get_state(config)
print(snapshot.next)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
print(snapshot.next)
