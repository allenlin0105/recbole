import csv
import random
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm


# DATASET = "amazon_toy"
DATASET = "ml-100k"

item_df = pd.read_csv(f"dataset/{DATASET}/{DATASET}.item", sep="\t")
inter_df = pd.read_csv(f"dataset/{DATASET}/{DATASET}.inter", sep="\t")

# title_key = "title:token"
title_key = "movie_title:token_seq"

item_dict = (
    item_df.astype("str")[["item_id:token", title_key]]
    .set_index("item_id:token")
    .to_dict()
)[title_key]

item_id_list = list(item_dict.keys())

inter_dict = (
    inter_df
    .sort_values("timestamp:float")
    .astype("str")
    .groupby("user_id:token")
    .agg({"item_id:token": list})
    .rename(columns={"item_id:token": "history_list"})
    .to_dict("index")
)

with open(Path("dataset", DATASET, "item_id2name.csv"), "w") as fp:
    writer = csv.writer(fp)
    for key, value in item_dict.items():
        writer.writerow([key, value])

min_seq_len = 5
with open(Path("dataset", DATASET, "history.csv"), "w") as fp:
    writer = csv.writer(fp)
    writer.writerow(["user_id", "history_items"])

    for user_id, history_dict in tqdm(inter_dict.items()):
        inter = history_dict["history_list"]
        if len(inter) < min_seq_len:
            continue

        writer.writerow([
            user_id, 
            [item_dict[item_id] for item_id in inter],
        ])
        