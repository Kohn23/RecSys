import pandas as pd


def reindex_item_from_processed_txt(input_file, output_file):
    """
        reindex items to [0,n_item-1]
    """
    all_items = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()[1:]  # 跳过用户 id
            for w in parts:
                item_id = int(w.split("|")[0])
                all_items.add(item_id)

    all_items = sorted(all_items)
    mapping = {old: new for new, old in enumerate(all_items)}

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            user, *items = line.strip().split()
            new_items = []
            for w in items:
                item_id, ts = w.split("|")
                new_id = mapping[int(item_id)]
                new_items.append(f"{new_id}|{ts}")
            f_out.write(user + " " + " ".join(new_items) + "\n")

    print(f"Done! Reindexed {len(all_items)} items, saved to {output_file}")


def reindex_user_from_ui_txt(input_file, output_file):
    """
        reindex user to [1,n_users]
    """

    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            user, item = line.strip().split()
            user = str(int(user) + 1)
            f_out.write(user + " " + item + "\n")

    print(f"Done! Reindexed and saved to {output_file}")


def reindex_item_from_ui_txt(input_file, output_file):
    """
        reindex items to [1,n_items]
    """

    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            user, item = line.strip().split()
            item = str(int(item) + 1)
            f_out.write(user + " " + item + "\n")

    print(f"Done! Reindexed and saved to {output_file}")



def reindex_consistent(df: pd.DataFrame, list_u):
    """
        Reindex domains with consistent item index
    """
    print(f"\n[info] Reindexing users and items ...")

    map_u = {u: idx for idx, u in enumerate(list_u)}

    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    items_a = sorted(df_a["item_id"].unique())
    items_b = sorted(df_b["item_id"].unique())

    map_i = {}
    for i, x in enumerate(items_a, 1):
        map_i[x] = i
    offset = len(items_a) + 1
    for i, x in enumerate(items_b):
        map_i[x] = offset + i  # domain B index consistent with A

    u_list = df["user_id"].tolist()
    i_list = df["item_id"].tolist()
    df['user_id'] = [map_u[u] for u in u_list]
    df['item_id'] = [map_i[i] for i in i_list]

    print(f"\tDone. Users: {len(map_u)}, Items A: {len(items_a)}, Items B: {len(items_b)}")
    return df, map_i, map_u


def reindex_independent(df: pd.DataFrame, list_u):
    """
        Reindex domains with independent item index
    """
    print(f"\n[info] Reindexing users and items ...")

    map_u = {u: idx for idx, u in enumerate(list_u)}
    u_list = df["user_id"].tolist()
    df['user_id'] = [map_u[u] for u in u_list]

    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    items_a = sorted(df_a["item_id"].unique())
    items_b = sorted(df_b["item_id"].unique())

    map_i_a = {i: idx for idx, i in enumerate(items_a)}
    map_i_b = {i: idx for idx, i in enumerate(items_b)}
    map_i = {**map_i_a, **map_i_b}

    i_list = df["item_id"].tolist()
    df['item_id'] = [map_i[i] for i in i_list]

    print(f"\tDone. Users: {len(map_u)}, Items A: {len(items_a)}, Items B: {len(items_b)}")
    return df, map_i_a, map_i_b, map_u