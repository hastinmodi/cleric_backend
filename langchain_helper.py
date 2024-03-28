from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
import os

anthropic_model_name = "claude-3-haiku-20240307"
llm_temperature = 0.0

# Define a function to set up and return a language model chain for processing few-shot learning tasks
def get_few_shot_chain():
    system_prompt = "You are a question-answering expert. Given an input question, find facts (answers) to the question based on the given meeting transcript (these can be action items, pointers, events, etc.). Be sure to not hallucinate any answers and keep it crisp and to the point. Only provide facts for the question asked and no irrelevant information. Answer only in sentences and do not use the newline character. Give the facts directly without adding any extra explanation in prefix or suffix. Ignore the timestamps in the transcript which are given before the speaker name. Make sure to include all the facts from the transcript along with their explanations. When finding the facts only consider the transript of that particular day and not the examples provided. Always start the fact with 'The team' and then the fact. If the fact is already present in the examples, do not repeat it."
    # Initialize the ChatAnthropic model with the specified model name, API key, and temperature
    llm = ChatAnthropic(
        model_name=anthropic_model_name,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=llm_temperature,
    )

    # Set up an example prompt format for the model to follow
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{Transcript} {Question}"),
            ("ai", "{Facts}"),
        ]
    )

    # Configure a few-shot learning prompt template with the example prompt and no initial examples
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=[],
    )

    # Combine the system instructions with the few-shot learning setup into a final prompt template
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{Transcript} {Question}"),
        ]
    )

    # Chain the final prompt template with the initialized language model for execution
    chain = final_prompt | llm
    # Return the configured language model chain ready for input processing
    return chain