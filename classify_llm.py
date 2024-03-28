from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import defaultdict
from datetime import timedelta
import os
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
from utils import (
    extract_date,
    str_to_date,
    date_to_str,
    merge_dicts,
    auto_approve_suggestions,
)

anthropic_model_name = "claude-3-haiku-20240307"
llm_temperature = 0.0
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to calculate the similarity between a new fact and a list of old facts
def fact_similarity(new_fact: str, old_facts: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    input_embedding = model.encode(new_fact)
    sentence_embeddings = model.encode(old_facts)
    similarities = [
        1 - cosine(input_embedding, embedding) for embedding in sentence_embeddings
    ]
    max_similarity_index = similarities.index(max(similarities))
    if max(similarities) > 0.3:
        most_similar_sentence = old_facts[max_similarity_index]
        return most_similar_sentence
    else:
        return ""

# Setup the system prompt for the large language model
system_prompt = """You are a linguist and I would like to take your opinion on whether the two sentences given below can be transformed into a compound sentence. Answer "Yes" or "No" only without any explanations. If Yes, state if the second sentence increases any information about the subject mentioned in the first sentence, respond with "Yes" or if it adds any new unknown information, respond with "No". Respond only in "Yes" or "No" without any additional explanations."""

# Initialize the Anthropic chat model with specified parameters
llm = ChatAnthropic(
    model_name=anthropic_model_name,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    temperature=llm_temperature,
)

# Define example and few-shot prompts for the model
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{Old Fact} {New Fact}"),
        ("ai", "{Classification}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=[],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{Old Fact} {New Fact}"),
    ]
)

chain = final_prompt | llm

# Initialize dictionaries to store classified facts
global classified_facts_add_all, classified_facts_remove_all, classified_facts_modify_all, classified_facts_remove_modify_prev, classified_facts_add_24, classified_facts_remove_24, classified_facts_modify_24

classified_facts_add_all = defaultdict(list)
classified_facts_remove_all = defaultdict(list)
classified_facts_modify_all = defaultdict(list)
classified_facts_remove_modify_prev = defaultdict(list)

classified_facts_add_24 = defaultdict(list)
classified_facts_remove_24 = defaultdict(list)
classified_facts_modify_24 = defaultdict(list)

# Function to classify facts based on similarity and additional information
def classify_facts(all_prev_facts: dict, prev_facts: dict, curr_facts: dict):
    for day, facts in curr_facts.items():
        for i in range(len(facts)):
            # If there are no previous facts, add current facts to 'add' category
            if not all_prev_facts:
                classified_facts_add_all[day] = facts
                classified_facts_add_24[day] = facts
                break
            # Find the most similar sentence from all the previous day's facts
            most_similar_sentence = fact_similarity(
                facts[i],
                all_prev_facts[date_to_str(str_to_date(day) - timedelta(days=1))],
            )
            # Classify the fact based on similarity and if no similar sentence is found, classify the fact as 'add'
            if most_similar_sentence == "":
                if facts[i] not in classified_facts_add_all[day]:
                    classified_facts_add_all[day].append(facts[i])

                if facts[i] not in classified_facts_add_24[day]:
                    classified_facts_add_24[day].append(facts[i])
                break

            # Invoke the chain with the old and new fact to classify the relationship
            fact_type = chain.invoke(
                {"Old Fact": most_similar_sentence, "New Fact": facts[i]}
            )

            # Process the model's output to determine classification
            fact_output = fact_type.content.split("\n")
            # If the model indicates the new fact adds information without altering the old fact
            if ("Yes" in fact_output[0]) and ("No" in fact_output[1]):
                if facts[i] not in classified_facts_add_all[day]:
                    classified_facts_add_all[day].append(facts[i])
                if any(
                    most_similar_sentence in s
                    for s in prev_facts[
                        date_to_str(str_to_date(day) - timedelta(days=1))
                    ]
                ):
                    if facts[i] not in classified_facts_add_24[day]:
                        classified_facts_add_24[day].append(facts[i])
            # If the model indicates the new fact enhances the old fact
            elif ("Yes" in fact_output[0]) and ("Yes" in fact_output[1]):

                if facts[i] not in classified_facts_modify_all[day]:
                    classified_facts_modify_all[day].append(facts[i])

                if (
                    most_similar_sentence
                    not in classified_facts_remove_modify_prev[day]
                ):
                    classified_facts_remove_modify_prev[day].append(
                        most_similar_sentence
                    )
                if any(
                    most_similar_sentence in s
                    for s in prev_facts[
                        date_to_str(str_to_date(day) - timedelta(days=1))
                    ]
                ):
                    if most_similar_sentence not in classified_facts_modify_24[day]:
                        classified_facts_modify_24[day].append(most_similar_sentence)
            # If the model does not confirm an additive or enhancing relationship, classify as 'remove'
            else:
                if facts[i] not in classified_facts_remove_all[day]:
                    classified_facts_remove_all[day].append(facts[i])
                if (
                    most_similar_sentence
                    not in classified_facts_remove_modify_prev[day]
                ):
                    classified_facts_remove_modify_prev[day].append(
                        most_similar_sentence
                    )
                if any(
                    most_similar_sentence in s
                    for s in prev_facts[
                        date_to_str(str_to_date(day) - timedelta(days=1))
                    ]
                ):
                    if most_similar_sentence not in classified_facts_remove_24[day]:
                        classified_facts_remove_24[day].append(most_similar_sentence)
    return (
        classified_facts_add_all,
        classified_facts_remove_all,
        classified_facts_modify_all,
        classified_facts_remove_modify_prev,
        classified_facts_add_24,
        classified_facts_remove_24,
        classified_facts_modify_24,
    )

# Function to get classified facts for all documents, potentially auto-approving them
def get_classified_facts(all_facts, urls, auto_approve):
    all_prev_facts = {}
    prev_facts = {}
    curr_facts = {}
    # Iterate through the URLs (documents) and classify the facts
    for i in range(len(urls)):
        key = extract_date(urls[i])
        if key not in all_facts:
            continue
        prev_key = None
        if i > 0 and extract_date(urls[i - 1]) in all_facts:
            prev_key = extract_date(urls[i - 1])
        curr_facts = {key: all_facts[key]}
        (
            add_all,
            remove_all,
            modify_all,
            remove_modify_prev,
            add_24,
            remove_24,
            modify_24,
        ) = classify_facts(all_prev_facts, prev_facts, curr_facts)

        # Aggregate facts for each document
        temp = []
        [temp.extend(l) for l in (add_all[key], remove_all[key], modify_all[key])]
        all_prev_facts[key] = temp
        if prev_key:
            all_prev_facts[key].extend(all_prev_facts[prev_key])
        prev_facts = curr_facts

    # If auto-approving is enabled, automatically process the classified facts
    if auto_approve:
        dates = []

        # Extract and sort dates from the URLs to determine document order
        for document in urls:
            dates.append(extract_date(document))

        dates.sort()

        # Auto-approve suggestions based on the classified facts
        add_facts_approved = classified_facts_add_all
        modify_facts_approved = merge_dicts(
            [classified_facts_remove_all, classified_facts_modify_all], dates
        )
        remove_facts_approved = classified_facts_remove_modify_prev
        # Apply auto-approval rules to the classified facts
        return auto_approve_suggestions(
            add_facts_approved,
            modify_facts_approved,
            remove_facts_approved,
            get_classified_facts_res(),
            dates,
        )
    # If not auto-approving, simply return the current state of classified facts
    return get_classified_facts_res()

# Helper function to consolidate the results of fact classification into a single response structure
def get_classified_facts_res():
    return {
        "classified_facts_add_all": classified_facts_add_all,
        "classified_facts_remove_all": classified_facts_remove_all,
        "classified_facts_modify_all": classified_facts_modify_all,
        "classified_facts_remove_modify_prev": classified_facts_remove_modify_prev,
        "classified_facts_add_24": classified_facts_add_24,
        "classified_facts_remove_24": classified_facts_remove_24,
        "classified_facts_modify_24": classified_facts_modify_24,
    }

# Function to generate a response based on the classified facts for a given set of documents
def get_response_classified_facts(documents):
    
    return {
        "classified_facts_add": classified_facts_add_all,
        "classified_facts_remove": classified_facts_remove_modify_prev,
        "classified_facts_modify": merge_dicts(
            [classified_facts_remove_all, classified_facts_modify_all], documents
        ),
    }

# Function to clear all the global dictionaries storing classified facts, resetting them for a new classification session
def reset_dicts():
    classified_facts_add_all.clear()
    classified_facts_remove_all.clear()
    classified_facts_modify_all.clear()
    classified_facts_remove_modify_prev.clear()
    classified_facts_add_24.clear()
    classified_facts_remove_24.clear()
    classified_facts_modify_24.clear()