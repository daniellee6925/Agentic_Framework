from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()  # allows resuming from interrupted point


class State(TypedDict):
    value: str


def node_a(state: State):
    print("Node A")
    return Command(goto="node_b", update={"value": state["value"] + "a"})


def node_b(state: State):
    print("Node B")

    human_response = interrupt("Do you want to go to C or D? Type C/D")

    print("Human Review wants to go to ", human_response)
    if human_response == "C" or human_response == "c":
        return Command(goto="node_c", update={"value": state["value"] + "b"})
    elif human_response == "D" or human_response == "d":
        return Command(goto="node_d", update={"value": state["value"] + "b"})


def node_c(state: State):
    print("Node C")
    return Command(goto=END, update={"value": state["value"] + "c"})


def node_d(state: State):
    print("Node D")
    return Command(goto=END, update={"value": state["value"] + "d"})


graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)

graph.set_entry_point("node_a")

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

initial_state = {"value": ""}

first_response = app.invoke(initial_state, config, stream_mode="updates")
second_response = app.invoke(Command(resume="D"), config=config, stream_mode="updates")
print(first_response)
print(second_response)
