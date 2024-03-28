from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from main_llm import find_facts
from flask_executor import Executor
from classify_llm import get_response_classified_facts
from utils import extract_date, auto_approve_suggestions

app = Flask(__name__, static_url_path="/static")
executor = Executor(app)

# Route to submit questions and documents for processing
@app.route("/submit_question_and_documents", methods=["POST"])
@cross_origin(origin="*")
def submit_question_and_documents():
    data = request.get_json()
    global question
    question = data["question"]
    documents = data["documents"].split(",")
    auto_approve = data["autoApprove"]

    executor.futures.pop("classified_facts")

    global facts
    facts = {}

    global dates
    dates = []

    # Extract and sort the dates from the documents
    for document in documents:
        dates.append(extract_date(document))

    dates.sort()

    # Find facts immediately if auto-approving, else submit to executor for background processing
    if auto_approve:
        facts = find_facts(documents, question, auto_approve)
    else:
        executor.submit_stored(
            "classified_facts", find_facts, documents, question, auto_approve
        )

    return jsonify({"message": "Documents and question submitted successfully."}), 200

# Route to get classified facts
@app.route("/get_classified_facts", methods=["GET"])
@cross_origin(origin="*")
def classify_facts():
    if not executor.futures.done("classified_facts"):
        return jsonify({"question": question, "status": "processing"}), 202

    future = executor.futures.pop("classified_facts")
    global classified_facts
    classified_facts = future.result()
    return (
        jsonify(
            {"status": "done", "classified_facts": get_response_classified_facts(dates)}
        ),
        200,
    )

# Route to submit approvals for modifications to the facts
@app.route("/submit_approvals", methods=["POST"])
@cross_origin(origins="*")
def submit_approvals():
    data = request.get_json()
    add_facts_approved = data["add_facts_approved"]
    modify_facts_approved = data["modify_facts_approved"]
    remove_facts_approved = data["remove_facts_approved"]

    global facts
    facts = auto_approve_suggestions(
        add_facts_approved, modify_facts_approved, remove_facts_approved, classified_facts, dates
    )

    return jsonify({"message": "Approvals submitted successfully."}), 200

# Route to get the current question and associated facts
@app.route("/get_question_and_facts", methods=["GET"])
@cross_origin(origin="*")
def get_question_and_facts():
    if not facts:
        if question:
            return jsonify({"question": question, "status": "processing"}), 202
        else:
            return jsonify({"status": "no facts to dispaly"}), 200
    return (
        jsonify({"question": question, "facts": facts, "status": "done"}),
        200,
    )

# Route to serve static files
@app.route("/<path:path>")
def static_file(path):
    return app.send_static_file(path), 200