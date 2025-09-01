import pandas as pd


def read_two_domains(path_a, path_b):
    df_a = pd.read_csv(path_a, sep=',', header=None,
                       names=['user_id', 'item_id', 'rating', 'timestamp']).drop(columns=['rating'])
    df_b = pd.read_csv(path_b, sep=',', header=None,
                       names=['user_id', 'item_id', 'rating', 'timestamp']).drop(columns=['rating'])

    # check duplicated items
    i_dup = set(df_a.item_id) & set(df_b.item_id)
    if i_dup:
        df_a = df_a[~df_a.item_id.isin(i_dup)]
        df_b = df_b[~df_b.item_id.isin(i_dup)]
        print(f'\t\tFound and deleted {len(i_dup)} duplicated item ID in both domains.')

    return df_a, df_b
