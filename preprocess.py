"""
    run preprocess preprocess functions here
"""
import os
import argparse
from utils.preprocess import *
from collections import defaultdict


def data_conversion():
    parser = argparse.ArgumentParser(description='CDSR Leave-One-Out Preprocess Script')

    # Training
    parser.add_argument('--preprocess', type=str, default='afo',
                        help='name of the dataset')
    parser.add_argument('--k_i', type=int, default=10,
                        help='least interactions for each users/items in both domains')
    parser.add_argument('--k_u', type=int, default=5,
                        help='least interactions for each users/items in each domain')
    parser.add_argument('--len_max', type=int, default=50,
                        help='least interactions for each users/items in each domain')
    args = parser.parse_args()

    (file_a, file_b) = MAPPING_FILE_NAME[args.data]
    path_a = f'dataset/origin_abxipp/Amazon-Dataset-Raw/{file_a}'
    path_b = f'dataset/origin_abxipp/Amazon-Dataset-Raw/{file_b}'
    file_a = file_a.split('.')[0].lower()
    file_b = file_b.split('.')[0].lower()
    save_a = f'dataset/{args.data}_{file_a}'
    save_b = f'dataset/{args.data}_{file_b}'
    if not os.path.exists(save_a):
        os.makedirs(save_a)
    if not os.path.exists(save_b):
        os.makedirs(save_b)

    if args.data in MAPPING_FILE_NAME.keys():
        print(f'\n[info] Start preprocessing "{args.data}" dataset...')
        df_a, df_b = read_two_domains(path_a, path_b)
    else:
        raise NotImplementedError(f'Selected dataset "{args.data}" is not supported.')

    df = filter_non_overlapped(df_a, df_b)
    df = filter_cold_item(df, args.k_i)
    df = trim_sequence(df, args.len_max)
    df, list_u = filter_mono_domain_user(df, args.k_u)
    df, *_ = reindex_independent(df, list_u)

    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    # # --------------TAT4SRec-----------------
    # save_as_txt_utsi(df_a, f'{save_a}/{args.preprocess}_{file_a}_TAT')
    # save_as_txt_utsi(df_b, f'{save_b}/{args.preprocess}_{file_b}_TAT')

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
    # # # save_as_jsons(dict_a, f'{save_a}/jsons')
    # # # save_as_jsons(dict_b, f'{save_b}/jsons')

    # -----------Bole inter saving--------------
    # save_as_inter(df_a, f'{save_a}/{args.preprocess}_{file_a}.inter')
    # save_as_inter(df_b, f'{save_b}/{args.preprocess}_{file_b}.inter')

    # --------General user-item txt saving-------
    save_as_txt_ui(df_a, f'{save_a}/{args.data}_{file_a}_ui.txt')
    save_as_txt_ui(df_b, f'{save_b}/{args.data}_{file_b}_ui.txt')


if __name__ == '__main__':
    # data_conversion()

    reindex_item_from_ui_txt(f'dataset/abh_beauty/abh_beauty_ui.txt',
                             f'dataset/abh_beauty/abh_beauty_ui_reindex.txt')
