import re
from datetime import datetime, timedelta
from collections import defaultdict

# Convert a string to a datetime object
def str_to_date(s):
    return datetime.strptime(s, "%Y-%m-%d")

# Convert a datetime object to a string
def date_to_str(d):
    return d.strftime("%Y-%m-%d")

# Extract a date from a filename assuming it contains a date in the format YYYYMMDD
def extract_date(file_name):
    match = re.search(r"\d{8}", file_name)
    if match:
        date_str = match.group()
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    return None

# Merge an array of dictionaries into a single dictionary, avoiding duplicate entries
def merge_dicts(dict_arr, documents):
    merged_dict = defaultdict(list)
    for dict in dict_arr:
        if dict:
            for key in dict.keys():
                if key in merged_dict:
                    for i in dict[key]:
                        if i not in merged_dict[key]:
                            merged_dict[key].append(i)
                else:
                    merged_dict[key] = dict[key]

    for document in documents:
        if document not in merged_dict:
            merged_dict[document] = []

    return merged_dict

# Accumulate facts over a series of dates, taking into account facts to remove
def accummulate_facts(facts, remove_facts_approved, dates):
    remove_facts = set()
    temp = defaultdict(list)
    remove_24 = defaultdict(list)

    for date in dates:
        if date in remove_facts_approved:
            remove_facts.update(remove_facts_approved[date])
        if date in facts:
            t = facts[date]
            prev_date = date_to_str(str_to_date(date) - timedelta(days=1))
            for i in temp[prev_date]:
                if i in remove_facts:
                    remove_24[date].append(i)
                if i not in t and i not in remove_facts:
                    t.append(i)
            temp[date] = t

    return (temp, remove_24)

# Automatically approve suggestions based on classified facts
def auto_approve_suggestions(
    add_facts_approved,
    modify_facts_approved,
    remove_facts_approved,
    classified_facts,
    dates,
):
    approved_facts = merge_dicts([add_facts_approved, modify_facts_approved], dates)

    # Filter the facts for each category by the approved facts
    for key in classified_facts["classified_facts_add_24"].keys():
        classified_facts["classified_facts_add_24"][key] = list(
            filter(
                lambda x: x in approved_facts[key] if key in approved_facts else False,
                classified_facts["classified_facts_add_24"][key],
            )
        )

    for key in classified_facts["classified_facts_remove_24"].keys():
        classified_facts["classified_facts_remove_24"][key] = list(
            filter(
                lambda x: (
                    x in remove_facts_approved[key]
                    if key in remove_facts_approved
                    else False
                ),
                classified_facts["classified_facts_remove_24"][key],
            )
        )

    for key in classified_facts["classified_facts_modify_24"].keys():
        classified_facts["classified_facts_modify_24"][key] = list(
            filter(
                lambda x: x in approved_facts[key] if key in approved_facts else False,
                classified_facts["classified_facts_modify_24"][key],
            )
        )

    # Accumulate all facts, considering the approved facts and those to be removed
    all_facts, remove_24 = accummulate_facts(
        approved_facts, remove_facts_approved, dates
    )

    # Global facts
    facts = {}
    facts["facts_by_day"] = approved_facts
    facts["all_facts"] = all_facts
    facts["add_24"] = classified_facts["classified_facts_add_24"]
    facts["remove_24"] = remove_24
    facts["modify_24"] = classified_facts["classified_facts_modify_24"]
    return facts