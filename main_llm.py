from langchain_helper import get_few_shot_chain
from langchain.document_loaders import UnstructuredURLLoader
from classify_llm import get_classified_facts, reset_dicts
from utils import extract_date

chain = get_few_shot_chain()

all_facts = {}

# Define a function to find facts in the given URLs with respect to a specific question
def find_facts(urls: list, question: str, auto_approve: bool):
    # Create a loader for unstructured data from the list of URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # Iterate through each URL's content
    for i in range(len(urls)):
        # Skip if the page content is empty or only contains whitespace
        if not data[i].page_content.strip():
            continue
        else:
            # Process the content with the chain to find facts related to the question and store them in the all_facts dictionary keyed by the date extracted from the URL
            all_facts[extract_date(urls[i])] = [
                facts.strip()
                for facts in chain.invoke(
                    {"Transcript": data[i].page_content, "Question": question}
                )
                .content[:-1]
                .split(".")
            ]
    reset_dicts()

    # Sort the dictionary of all facts by their keys (dates)
    keys = list(all_facts.keys())
    keys.sort()
    sorted_dict = {i: all_facts[i] for i in keys}

    # Classify the sorted facts based on the URLs and whether auto-approval is enabled
    return get_classified_facts(sorted_dict, urls, auto_approve)