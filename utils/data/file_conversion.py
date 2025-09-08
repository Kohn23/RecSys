"""
    data preprocessing functions
"""

import pandas as pd
import json
import os


def txt_to_inter(input_file, output_file):
    """
    input_file: .txt
    0 4559|1357430400 27301|1357430400 12890|1373241600 ...
    1 5778|1385337600 5985|1385337600 8366|1385337600 ...

    output_file: .inter
    user_id:token	item_id:token	timestamp:float
    0	4559	1357430400
    0	27301	1357430400
    1	5778	1385337600
    1	5985	1385337600
    ...
    """

    rows = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            user_id = parts[0]

            for interaction in parts[1:]:
                if '|' in interaction:
                    item_id, timestamp = interaction.split('|')
                    rows.append({
                        'user_id:token': user_id,
                        'item_id:token': item_id,
                        'timestamp:float': timestamp
                    })

    df = pd.DataFrame(rows)

    df.to_csv(output_file, sep='\t', index=False)
    print(f"output: {output_file}")


def txt_to_jsons(input_file, output_dir):
    """
        Conversion to MGT format

    input_file: .txt
    0 4559|1357430400 27301|1357430400 12890|1373241600 ...
    1 5778|1385337600 5985|1385337600 8366|1385337600 ...

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

    rows = {}
    all_items = set()
    all_users = set()

    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            user_id = parts[0]
            all_users.add(user_id)
            interactions = []

            for interaction in parts[1:]:
                if '|' in interaction:
                    item_id, _ = interaction.split('|')
                    interactions.append(item_id)
                    all_items.add(item_id)

            rows[user_id] = interactions

    user_map = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
    item_map = {item_id: idx for idx, item_id in enumerate(sorted(all_items))}

    train_data = {}
    test_data = {}
    val_data = {}

    for user_id, items in rows.items():
        if len(items) < 3:
            continue

        mapped_items = [item_map[item] for item in items]

        train_data[user_map[user_id]] = mapped_items[:-2]

        val_data[user_map[user_id]] = [mapped_items[-2]]

        test_data[user_map[user_id]] = [mapped_items[-1]]

    # 保存结果到文件
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f)

    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)

    with open(os.path.join(output_dir, 'smap.json'), 'w') as f:
        json.dump(item_map, f)

    with open(os.path.join(output_dir, 'umap.json'), 'w') as f:
        json.dump(user_map, f)

    print(f"Successfully saved to {output_dir}")
