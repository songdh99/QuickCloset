import pandas as pd
import json
import gc


with open('./data.json', 'r') as f:
    json_data = f.readlines()
    for line in json_data:
        print(line)
print(json_data)
