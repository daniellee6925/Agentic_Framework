from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

memory = MemorySaver()

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model="llama3-8b-8192")
llm_with_tools = llm.bind_tools(tools=tools)


class BasicState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tools_router(state: BasicState):
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_calls")) and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

app = graph.compile(checkpointer=memory, interrupt_before=["tool_node"])

config = {"configurable": {"thread_id": "1"}}

events = app.stream(
    {"messages": [HumanMessage(content="Search the current wheather in fremont?")]},
    config=config,
    stream_mode="updates",
)

for event in events:
    print(event)

snapshot = app.get_state(config=config)
print(snapshot.next)
