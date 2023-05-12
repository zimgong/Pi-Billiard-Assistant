import json

dictionary = {
    'lower': 120,
    'sensitivity': 175
}

json_object = json.dumps(dictionary, indent = 4)

with open("cali.json", "w") as outfile:
    outfile.write(json_object)