import pandas as pd
import json
from os.path import join
from tqdm import tqdm


def filter_non_overlapped(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """
        filter out non-overlapped user from both domains
    """
    print(f'\n[info] Retaining users have interactions on both domains only...')

    u_a, u_b = set(df_a['user_id'].tolist()), set(df_b['user_id'].tolist())
    u_ab = u_a.intersection(u_b)

    df_a = df_a[df_a['user_id'].isin(u_ab)]
    df_b = df_b[df_b['user_id'].isin(u_ab)]

    df_a.insert(3, 'domain', [0] * df_a.shape[0], True)
    df_b.insert(3, 'domain', [1] * df_b.shape[0], True)

    df = pd.concat([df_a, df_b]).sort_values(['user_id', 'timestamp'])

    print(f'\t\tDataset remains {len(df["user_id"].unique())} users, '
          f'{len(df_a["item_id"].unique())} A item, {df_a.shape[0]} A interactions, '
          f'{len(df_b["item_id"].unique())} B item and {df_b.shape[0]} B interactions.')

    return df


def filter_cold_item(df: pd.DataFrame, k_i):
    """
        filter out items with less than k_i interactions
    """
    print(f'\n[info] Filtering out cold-start items less than {k_i} interactions ...')

    cnt_i = df['item_id'].value_counts()
    i_k = cnt_i[cnt_i >= k_i].index

    df = df[df['item_id'].isin(i_k)]

    print(f'\t\tDataset remains {len(df["user_id"].unique())} users and {len(df["item_id"].unique())} items ')
    return df


def trim_sequence(df: pd.DataFrame, len_max):
    """
        trim sequences exceeding the maximum interaction length len_max
    """
    print(f"\n[info] Trimming to {len_max} per user")
    df = df.groupby('user_id').tail(len_max)
    print(f'\t\tDataset remains {len(df["user_id"].unique())} users and {len(df["item_id"].unique())} items ')

    return df


def filter_mono_domain_user(df: pd.DataFrame, k_u):
    """
        filter out user with less than k_u interactions per domain
    """
    print(f"\n[info] filtering mono domain user with less than {k_u} interactions each")

    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    cnt_u_a = df_a['user_id'].value_counts()
    cnt_u_b = df_b['user_id'].value_counts()

    u_k_a = cnt_u_a[cnt_u_a >= k_u].index
    u_k_b = cnt_u_b[cnt_u_b >= k_u].index
    list_u = sorted(set(u_k_a).intersection(set(u_k_b)))

    df = df[df['user_id'].isin(list_u)]

    print(f'\t\tDataset remains {len(df["user_id"].unique())} users, '
          f'{len(df_a["item_id"].unique())} A items, {df_a.shape[0]} A interactions, '
          f'{len(df_b["item_id"].unique())} B items and {df_b.shape[0]} B interactions.')
    return df, list_u

