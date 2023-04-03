import json
import sys

def merge_json_dicts(dict1, dict2, path):

    max_id = max(dict1.values())

    for word, id in dict2.items():
        if word not in dict1:
            max_id += 1
            dict1[word] = max_id

    # Write the merged dictionary to a new file
    with open(path+"/vocabulary.json", "w") as f:
        json.dump(dict1, f)

# Load the dictionaries from the files
dict1_file = sys.argv[1]
dict2_file = sys.argv[2]
path = sys.argv[3]

with open(dict1_file, "r") as f:
    dict1 = json.load(f)

with open(dict2_file, "r") as f:
    dict2 = json.load(f)

# Merge the dictionaries
merge_json_dicts(dict1, dict2, path=path)