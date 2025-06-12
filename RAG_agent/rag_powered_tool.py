from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

embedding_function = OpenAIEmbeddings()

docs = [
    Document(page_content="some content here", metadata={"source": "about.txt"}),
    Document(page_content="some content here", metadata={"source": "hours.txt"}),
    Document(page_content="some content here", metadata={"source": "membership.txt"}),
    Document(page_content="some content here", metadata={"source": "classes.txt"}),
    Document(page_content="some content here", metadata={"source": "trainers.txt"}),
    Document(page_content="some content here", metadata={"source": "facilities.txt"}),
]

# chroma is an open source vector DB
# Indexes vectors and lets you search for similar content
db = Chroma.from_documents(docs, embedding_function)

# MMR: Maximal Marginal Relevance (balances relavence and diversity), k:3 -> top 3 results
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever, "retriever_tool", "Information about topic here"
)


@tool
def off_topic():
    """Catch all questions not related to topic"""
    return "Forbidden - do  not respond to the user"


tools = [retriever_tool, off_topic]


class AgentState(TypedDict):
    # list of chat messages that uses 'add_messages' to append messages
    messages: Annotated[Sequence[BaseMessage], add_messages]


def agent(state):
    messages = state["messages"]
    model = ChatOpenAI()
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": response}


# must be either 'tools' or END
def should_continue(state) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)

tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()
