import json
import os

if not os.path.exists("training_data"):
    os.makedirs("training_data")


data_sources = ["/mnt/afs/zhangyaolun/safe_model/masrl/training_data/designer_1225.jsonl"
]
designer_data = []
executor_data = []
for data_source in data_sources:
    with open(data_source, "r") as f:
        if "designer" in data_source:
            data = []
            for line in f:
                assistant_response = json.loads(line.strip())["conversations"][2]["value"]
                if len(assistant_response) < 100:
                    continue    
                data.append(json.loads(line.strip()))   
            designer_data.extend(data)
        elif "executors" in data_source:
            data = []
            for line in f:
                data.append(json.loads(line.strip()))
            executor_data.extend(data)

with open("training_data/designer_1226.jsonl", "w") as f:
    for data in designer_data:
        f.write(json.dumps(data) + "\n")

with open("training_data/executors_1225.jsonl", "w") as f:
    for data in executor_data:
        f.write(json.dumps(data) + "\n")

print(len(designer_data))
print(len(executor_data))