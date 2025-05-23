from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain
import os

load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state):
    # sends a list of chat messages 'state' and appends models reply
    return generation_chain.invoke({"messages": state})


def reflect_node(state):
    response = reflection_chain.invoke({"message": state})
    # set reflection message as if it was from a human
    return [HumanMessage(content=response.content)]


# create nodes
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)


def should_continue(state):
    if len(state) > 4:
        return END
    return REFLECT


# create edges
graph.add_conditional_edges(
    GENERATE, should_continue, path_map={REFLECT: REFLECT, END: END}
)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()
