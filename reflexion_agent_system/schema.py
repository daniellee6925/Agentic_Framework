from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer Questions"""

    answer: str = Field(description="~250 word detailed answer to the question.")
    searfch_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your answer."
    )
    reflection: str = Field(description="your reflection on the initial answer")


class ReviseAnswer(BaseModel):
    """Revise your original answer to the question"""

    references: List[str] = Field(
        description="Citations motivating your updated answer"
    )
