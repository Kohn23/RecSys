import os
import pandas as pd


def _check_duplicated_items(df_a: pd.DataFrame, df_b: pd.DataFrame):
    i_dup = set(df_a.item_id) & set(df_b.item_id)
    if i_dup:
        df_a = df_a[~df_a.item_id.isin(i_dup)]
        df_b = df_b[~df_b.item_id.isin(i_dup)]
        print(f'\t\tFound and deleted {len(i_dup)} duplicated item ID in both domains.')
    return df_a, df_b


def _detect_format(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.csv':
        return 'csv'
    elif ext in ['.jsonl', '.json']:
        return 'jsonl'
    else:
        return 'csv'


def _read_single(file: str,
                file_type: str = None,
                select_cols: list = None,
                rename_cols: dict = None):
    if file_type == 'jsonl' or file_type == 'json':
        df = pd.read_json(file, lines=True)
    else:
        df = pd.read_csv(file, sep=',')

    if rename_cols is not None:
        df = df.rename(columns=rename_cols)

    if select_cols is None:
        select_cols = ['user_id', 'item_id', 'rating', 'timestamp']

    return df[select_cols]


def _read_double(file_a: str, file_b: str = None,
                file_type: str = None,
                select_cols: list = None,
                rename_cols: dict = None):
    """
        read domains and make cross-domain
    """

    df_a = _read_single(file_a, file_type, select_cols, rename_cols)
    print(df_a.head())

    if file_b is None:
        return df_a, None

    df_b = _read_single(file_b, file_type, select_cols, rename_cols)

    return _check_duplicated_items(df_a, df_b)


def read_raw(file_a: str, file_b: str = None,
             file_type: str = None,
             select_cols: list = None,
             rename_cols: dict = None):

    if file_type is None:
        file_type = _detect_format(file_a)

    return _read_double(file_a, file_b, file_type, select_cols, rename_cols)


