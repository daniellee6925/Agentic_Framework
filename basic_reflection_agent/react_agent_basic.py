from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

search_tool = TavilySearchResults(search_depth="basic")


@tool  # let langchain know it's a tool
def get_system_time(format: str = "%Y-%m %d %H:%M:%S"):
    """Returns the current date and time"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = initialize_agent(
    tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True
)

result = llm.invoke("Give me a fact about cats")

print(result)
