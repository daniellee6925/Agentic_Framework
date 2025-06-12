from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

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


qa_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x)), "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

qa_chain.invoke("Ask question here")
