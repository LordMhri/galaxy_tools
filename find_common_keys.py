import json



with open("galaxy_tools.json","r") as f:
    common_keys = None


    for i,line in enumerate(f):
        records = json.loads(line)

        print(records)