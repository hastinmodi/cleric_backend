# Fact-Finding and Classification System

This system is designed to process a collection of documents with the purpose of extracting and classifying facts in response to user-submitted questions. It leverages the capabilities of large language models for natural language processing to provide a structured and efficient way of handling information extracted from unstructured data sources.

## Features

- **Question Submission**: Users can submit questions along with a set of documents. The system processes these documents to find answers to the submitted questions.
- **Fact Finding**: Utilizes a large language model to extract relevant facts from a collection of documents.
- **Fact Classification**: Classifies the extracted facts into categories such as new facts, modified facts, and facts to be removed, based on their relevance and information content.
- **Auto-Approval**: For efficiency, the system can automatically approve classifications based on predefined criteria.
- **REST API**: Provides a RESTful interface for submitting questions, documents, and for retrieving processed information.

## System Architecture

The system is built using Flask, a lightweight WSGI web application framework in Python, and is designed to be both scalable and modular. The architecture includes several key components:

- **Flask Application (`app.py`)**: The entry point of the application, handling HTTP requests and responses.
- **Language Model Chain (`langchain_helper.py`)**: Manages the interaction with large language models for processing natural language.
- **Fact Finder (`main_llm.py`)**: Implements the logic for finding facts within documents based on user questions.
- **Fact Classifier (`classify_llm.py`)**: Responsible for classifying extracted facts into various categories.
- **Utilities (`utils.py`)**: Provides utility functions for date manipulation, text processing, and more.

## Setup and Installation

Ensure you have Python 3.8 or later installed on your system. Follow these steps to set up the project locally:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
gunicorn -k eventlet -b 0.0.0.0:5000 --timeout 0 app:app
```

The application will start running on localhost at the port specified (default is 5000).

The application is also hosted on AWS EC2 - https://cleric.bhagavadgita.tech/

## Usage
The system exposes several endpoints for interaction:

1. Submit Question and Documents: POST /submit_question_and_documents
2. Get Classified Facts: GET /get_classified_facts
3. Submit Approvals: POST /submit_approvals
4. Get Question and Facts: GET /get_question_and_facts

