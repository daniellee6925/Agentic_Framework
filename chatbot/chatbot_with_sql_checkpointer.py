from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)

llm = ChatGroq(model="llama3-8b-8192")


class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):
    return {"messages": llm.invoke(state["messages"])}


graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 1}}

"""
response1 = app.invoke(
    {"messages": HumanMessage(content="Hi, I'm Daniel")}, config=config
)

response2 = app.invoke(
    {"messages": HumanMessage(content="What's my name?")}, config=config
)

print(response1)
print("\n")
print(response2)
"""
while True:
    user_input = input("User: ")
    if user_input in ["exit", "end"]:
        break
    else:
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        print("AI: " + result["messages"][-1].content)
