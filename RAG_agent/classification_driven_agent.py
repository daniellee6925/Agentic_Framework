from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.schema import Document
from langgraph.graph import add_messages, StateGraph, END
from pydantic import BaseModel, Field

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

retriever.invoke("Ask question here")


template = """Answer the question based only on the following conext: {context} Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)


llm = ChatOpenAI(model="gpt-4o")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x)), "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)


"""Build Agent Flow"""


class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: str


class GradeQeustion(BaseModel):
    """Boolean value to check wheter a question is related othe the topic"""

    score: str = Field(
        description="Question is aobut topic? If yes-> 'YES' if not -> 'NO'"
    )


def question_classifier(state: AgentState):
    question = state["messages"][-1].content
    system = """You are a classifier that determines whether a user's question is about one of the following topics
    1. topic one
    2. topic two
    3. topic three
    
    If the quetion IS about any of these  topics, respond with 'Yes'. Otherwise, respond 'No'
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [("System", system), ("human", "User question: {question}")]
    )

    llm = ChatOpenAI(model="gpt-4o")
    structured_llm = llm.with_structured_output(GradeQeustion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    state["on_topic"] = result.score

    return state


def on_topic_router(state: AgentState):
    on_topic = state["on_topic"]
    if on_topic.lower() == "yes":
        return "on_topic"
    return "off_topic"


def retrieve(state: AgentState):
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state


def generate_answer(state: AgentState):
    question = state["messages"][-1].content
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    state["messages"].append(generation)


def off_topic_respone(state: AgentState):
    state["messages"].append(
        AIMessage(content="I'm sorry, I cannot answer that question")
    )
    return state


workflow = StateGraph(AgentState)

workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_respone)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_answer", generate_answer)

workflow.add_conditional_edges(
    "topic_decision",
    on_topic_router,
    {"on_topic": "retrieve", "off_topic": "off_topic_response"},
)

workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)

workflow.set_entry_point("topic_decision")

app = workflow.compile()
