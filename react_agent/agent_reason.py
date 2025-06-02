from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain import hub
from dotenv import load_dotenv
import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-4")
search_tool = TavilySearchResults(search_depth="basic")


@tool
def get_system_time(format: str = "%Y-%m %d %H:%M:%S"):
    """Returns the current date and time"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

react_prmopt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prmopt)
