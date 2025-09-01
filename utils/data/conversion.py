"""
    data preprocessing functions
"""

import pandas as pd


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


def df_to_inter(input: pd.DataFrame, output_file):
    """
    input: pd.DataFrame with the right format (drop rating)

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
