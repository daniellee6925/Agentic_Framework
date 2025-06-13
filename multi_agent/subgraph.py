from typing import TypedDict, Annotated, Dict
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

load_dotenv()


class ChildState(TypedDict):
    messages: Annotated[list, add_messages]


search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model="llama3-8b-8192")

llm_with_tools = llm.bind_tools(tools=tools)


def agent(state: ChildState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tools_router(state: ChildState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


tool_node = ToolNode(tools=tools)

subgraph = StateGraph(ChildState)
subgraph.add_node("agent", agent)
subgraph.add_node("tool_node", tool_node)
subgraph.set_entry_point("agent")

subgraph.add_conditional_edges("agent", tools_router)
subgraph.add_edge("tool_node", "agent")

search_app = subgraph.compile()


"""shared schema"""


# Define parent graph with the same schema
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]


# create parent graph
parent_graph = StateGraph(ParentState)

# add the subgraph as a node
parent_graph.add_node("search_agent", search_app)

# connect the flow
parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)

parent_app = parent_graph.compile()

# run the parent graph
result = parent_app.invoke(
    {"messages": [HumanMessage(content="Search for the weather in fremont")]}
)


"""Different schema"""


# different schema from the child state
class QueryState(TypedDict):
    query: str
    response: str


# function to invoke subgraph
def search_agent(state: QueryState) -> Dict:
    # transform the parent schema to subgraph schema
    subgraph_input = {"messages": [HumanMessage(content=state["query"])]}

    # invoke the subgraph
    subgraph_result = search_app.invoke(subgraph_input)

    # transform the response back to parent schema
    assistant_message = subgraph_result["messages"][-1]
    return {"response": assistant_message.content}


# create parent graph
parent_graph = StateGraph(ParentState)

# add transformation node that invokes subgraph
parent_graph.add_node("search_agent", search_agent)

# connect the flow
parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)


parent_app = parent_graph.compile()

# run the parent graph
result = parent_app.invoke({"query": "Search for the weather in fremont"})
