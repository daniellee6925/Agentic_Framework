from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

# actor agent prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher. Current time: {time}
            1. {first_instruction}
            2. Reflect and critique your anser. Be sever to maximize improvement
            3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the users's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)  # prefill time (system info)

# responder ---------------------------------------------------------------
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

llm = ChatOpenAI(model="gpt-4o")

# chain components together. Get user input -> fill it in to the prompt -> send to llm -> parse output using pydantic
# bind tools -> attach tools to llm
first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

validator = PydanticToolsParser(tools=[AnswerQuestion])

# revisor ---------------------------------------------------------------
revise_instruction = """Revise your previous answer using the new information.
- you should use the previous critique to add important information to your answer.
You MUST include numerical citations in your revised answer to ensure it can be verified
- Add a "References" Section to the bottom of your answer (doesn't count in word limit) in the form of:
-[1] www.example.com
-[2] www.example.com
-You should use the previous critique to remove superfluous information from your answer and make sure it is less than 250 words."""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# invoke the chain
response = first_responder_chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write me a blog post on how small businesses can leverage AI to grow"
            )
        ]
    }
)

print(response)
