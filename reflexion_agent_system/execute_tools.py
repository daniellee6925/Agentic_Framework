import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults

# create search tool
tavily_tool = TavilySearchResults(max_results=5)

# function to execute search queries from Answer questions tool


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]

    # extract tool call from last Ai message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    # process the AnswerQuestion or ReviseAnswer tool calls by extracting search queries
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            # execute each search query using tavily search tool
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result

        # create a tool message with results
        tool_messages.append(
            ToolMessage(content=json.dump(query_results), tool_call_id=call_id)
        )
    return tool_messages
