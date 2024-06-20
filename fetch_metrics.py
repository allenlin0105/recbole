import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="")
args = parser.parse_args()
run_paths = [args.run]

import os

target_metrics = [
    "test/ndcg@1",
    "test/ndcg@5",
    "test/ndcg@10",
    "test/ndcg@20",
    "test/hit@1",
    "test/hit@5",
    "test/hit@10",
    "test/hit@20",
]
metric2scores = {metric: [] for metric in target_metrics}

import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
# run_path = "allenlin/recbole/a7csi24i"

import numpy as np

"""
Test seed:
1234
5678
123
7654
777
"""

## DuoRec
# run_paths = [
#     "allenlin/recbole/mjawnlei",
#     "allenlin/recbole/t0oe3j0l",
#     "allenlin/recbole/3xmct3br",
#     "allenlin/recbole/fk2euae9",
#     "allenlin/recbole/vdrwu975",
# ]

## DuoRec + MSELoss 
# run_paths = [
#     "allenlin/recbole/8aywsgx9",
#     "allenlin/recbole/12ksgkxq",
#     "allenlin/recbole/3sf4oc4i",
#     "allenlin/recbole/f281x0ec",
#     "allenlin/recbole/eij02dtp",
# ]

## DuoRec + MSELoss + PCA
# run_paths = [
#     "allenlin/recbole/d2zsk3fd",
#     "allenlin/recbole/9y6x0qe7",
#     "allenlin/recbole/1gmuqik3",
#     "allenlin/recbole/6uk5ewxs",
#     "allenlin/recbole/t1u9nc1j",
# ]

## DuoRec + PCA + KLDiv
# run_paths = [
#     "allenlin/recbole/w2ui5j9c",
#     "allenlin/recbole/a85w4jdm",
#     "allenlin/recbole/paf1259o",
#     "allenlin/recbole/errjpspe",
#     "allenlin/recbole/jtn6rvpz",
# ]

for run_path in run_paths:
    run = api.run(run_path)

    log_file = 'output.log'
    file = run.file(log_file)
    file.download(replace=True)

    with open("output.log", "r") as fp:
        lines = [line for line in fp]

    val_line = lines[-2]
    test_line = lines[-1]

    def parse_line(split, line):
        import json

        index = line.find("{")
        # print(line)
        # print(line[index:].replace("\'", "\""))
        result = json.loads(line[index:].replace("\'", "\""))

        for metric, value in result.items():
            key = f"{split}/{metric}"
            if key in metric2scores:
                metric2scores[key].append(float(value))
            if "ndcg" in metric or "hit" in metric:
                print(f"{split}/{metric}: {value}")

    # parse_line("val", val_line)
    parse_line("test", test_line)

    os.remove(log_file)

if len(run_paths) > 1:
    for metric, scores in metric2scores.items():
        print(f"{metric}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
exit()
# parse metrics
metrics_dataframe = run.history()

cols = metrics_dataframe.columns
metrics, indices = [], []
metric_max_len = 0
for i, col in enumerate(cols):
    # if col[:4] == "val/" or col[:5] == "test/":
    if "ndcg" in col or "hit" in col:
        metrics.append(col)
        indices.append(i)
        metric_max_len = max(metric_max_len, len(col))

target_metric = "val/recall@20"
target_row = metrics_dataframe[target_metric].idxmax()

print(f"Max {target_metric} happens at epoch {target_row}")

values = metrics_dataframe.iloc[target_row][indices].values.tolist()

metrics_values = [(metric, value) for metric, value in zip(metrics, values)]
metrics_values = sorted(metrics_values)
for metric, value in metrics_values:
    pad = " " * (metric_max_len - len(metric))
    print(f"{metric}:{pad} {value}")
