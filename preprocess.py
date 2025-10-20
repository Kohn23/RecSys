"""
    run preprocess preprocess functions here
"""
import os
import argparse
from utils.preprocess import *
from collections import defaultdict


def process_raw():
    parser = argparse.ArgumentParser(description='CDSR Leave-One-Out Preprocess Script')

    parser.add_argument('--prefix', type=str, default='amb',
                        help='name of the dataset')
    parser.add_argument('--k_i', type=int, default=10,
                        help='least interactions for each users/items in both domains')
    parser.add_argument('--k_u', type=int, default=5,
                        help='least interactions for each users/items in each domain')
    parser.add_argument('--len_max', type=int, default=50,
                        help='max interactions for each users/items in each domain')
    args = parser.parse_args()

    (file_a, file_b) = MAPPING_FILE_NAME[args.prefix]

    path_a = f'dataset/raw/amazon-reviews-2018/{file_a}'
    path_b = f'dataset/raw/amazon-reviews-2018/{file_b}'

    file_a = file_a.split('.')[0].lower()
    file_b = file_b.split('.')[0].lower()
    dir_a = f'dataset/{args.prefix}_{file_a}'
    dir_b = f'dataset/{args.prefix}_{file_b}'

    if not os.path.exists(dir_a):
        os.makedirs(dir_a)
    if not os.path.exists(dir_b):
        os.makedirs(dir_b)

    if args.prefix in MAPPING_FILE_NAME.keys():
        print(f'\n[info] Start preprocessing "{args.prefix}" dataset...')

        col_names = ['user_id', 'item_id', 'rating', 'timestamp']
        rename_cols = {'parent_asin':'item_id'}
        select_cols = ['user_id', 'item_id', 'timestamp']

        df_a, df_b = read_raw(path_a, path_b, name_cols=col_names, select_cols=select_cols, rename_cols=rename_cols)
    else:
        raise NotImplementedError(f'Selected dataset "{args.prefix}" is not supported.')

    df = filter_non_overlapped(df_a, df_b)
    df = filter_cold_item(df, args.k_i)
    df = trim_sequence(df, args.len_max)
    df, list_u = filter_mono_domain_user(df, args.k_u)
    df, *_ = reindex_independent(df, list_u)

    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    # # --------------TAT4SRec-----------------
    # save_as_txt_utsi(df_a, f'{dir_a}/{args.prefix}_{file_a}_TAT')
    # save_as_txt_utsi(df_b, f'{dir_b}/{args.prefix}_{file_b}_TAT')

    # -----------MGT json saving--------------
    # dict_a = defaultdict(list)
    # dict_b = defaultdict(list)
    #
    # # for _, row in df_a.iterrows():
    # #     u = row.iloc[0]
    # #     i = row.iloc[1]
    # #     dict_a[u].append(i)
    # #
    # # for _, row in df_b.iterrows():
    # #     u = row.iloc[0]
    # #     i = row.iloc[1]
    # #     dict_b[u].append(i)
    # #
    # # # save_as_jsons(dict_a, f'{dir_a}/jsons')
    # # # save_as_jsons(dict_b, f'{dir_b}/jsons')

    # -----------Bole inter saving--------------
    save_as_inter(df_a, f'{dir_a}/{args.prefix}_{file_a}.inter')
    save_as_inter(df_b, f'{dir_b}/{args.prefix}_{file_b}.inter')

    # --------General user-item txt saving-------
    # save_as_txt_ui(df_a, f'{dir_a}/{args.prefix}_{file_a}_ui.txt')
    # save_as_txt_ui(df_b, f'{dir_b}/{args.prefix}_{file_b}_ui.txt')


if __name__ == '__main__':
    process_raw()


