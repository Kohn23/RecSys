import os
import json
import pandas as pd
from collections import defaultdict


def save_as_inter(input: pd.DataFrame, output_file):
    """
    input: pd.DataFrame with the .inter format (drop rating)

    output_file: .inter
    user_id:token	item_id:token	timestamp:float
    0	4559	1357430400
    0	27301	1357430400
    1	5778	1385337600
    1	5985	1385337600
    ...
    """

    required_cols = ['user_id', 'item_id', 'timestamp']
    for col in required_cols:
        if col not in input.columns:
            raise ValueError(f"missing col: {col}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for row in input.itertuples(index=False):
            f.write(f"{row.user_id}\t{row.item_id}\t{row.timestamp}\n")


def save_as_txt_ui(df: pd.DataFrame, output_file):
    """
    input: pd.DataFrame with the .inter format (drop rating)

    output_file: .txt
    0 4559
    0 27301
    1 5778
    1 5985
    ...
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.user_id} {row.item_id}\n")


def save_as_txt_utsi(df: pd.DataFrame, file_path):
    """
    input: pd.DataFrame with the .inter format (drop rating)

    output_file: .txt
    0 3555000 4559
    0 3557200 27301
    1 3554900 5778
    1 3555200 5985
    ...
    """
    output_file = f'{file_path}.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.user_id} {row.timestamp} {row.item_id}\n")


def save_as_jsons(data: dict, output_dir):
    """
        Conversion to MGT format

    dict:
     {
        0: [5045, 4097, 11699, ... ],
        1: [10396, 11365, 3594, ... ],
        ...
    }


    output_dir: .jsons
    train.json:
        {
            "0":[4559, 27301, 12890 ...]
            "1":[5778, 5985, 8366 ...]
            ...
        }
    test.json:
        {
            "0":[<last_item_idx>]
            "1":[<last_item_idx>]
            ...
        }
    val.json:
        {
            "0":[<last_but_second_item_idx>]
            "1":[<last_but_second_item_idx>]
            ...
        }
    smap.json:{"0":0,"1":1, ... }
    umap.json:{"0":0,"1":1, ... }
    """

    os.makedirs(output_dir, exist_ok=True)
    user_ids = sorted(data.keys())
    item_ids = sorted(set(item for items in data.values() for item in items))

    umap = {str(user_id): idx for idx, user_id in enumerate(user_ids)}
    smap = {str(item_id): idx for idx, item_id in enumerate(item_ids)}

    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)

    for user_id, items in data.items():
        if len(items) < 3:
            continue

        user_str = str(user_id)

        train_items = [int(item) for item in items[:-2]]
        val_item = int(items[-2])
        test_item = int(items[-1])

        train_data[user_str].extend(train_items)
        val_data[user_str].append(val_item)
        test_data[user_str].append(test_item)

    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)

    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f)

    with open(os.path.join(output_dir, 'smap.json'), 'w') as f:
        json.dump(smap, f)

    with open(os.path.join(output_dir, 'umap.json'), 'w') as f:
        json.dump(umap, f)