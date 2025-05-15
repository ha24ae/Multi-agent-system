from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import Annotated

from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool



load_dotenv()
llm_name = "claude-3-haiku-20240307"
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name, temperature=1)
memory = MemorySaver()

#it is here where we define our stat where we can customise it by adding additional fields
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

#now populate the state key inside of the human_assistance tool.This allows the human to review the information before it is stored in the state.
#Command to issue a state update from inside the tool

@tool
def human_assistance(name:str, birthday:str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Request assistance from a human"""
    human_response = interrupt(
        {
        "question": "Is this correct?",
        "name":name,
        "birthday":birthday,
        },
    )
    #if the information if correct update the state as it is
    if human_response.get("correct","").lower().startwith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "correct"

    #otherwise lets recieve info from human reviewer
    else:
        verified_name =human_response.get("name",name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id= tool_call_id)],
    }
    return Command(update=state_update)






