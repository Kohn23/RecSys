from utils.preprocess.read import *
from utils.preprocess.filtering import *
from utils.preprocess.file_conversion import *
from utils.preprocess.save import *
from utils.preprocess.count import *
from utils.preprocess.reindex import *

IDX_PAD = 0

MAPPING_FILE_NAME = {
    'afo': ('Food.csv', 'Office.csv'),
    'abh': ('Beauty.csv', 'Health.csv'),
    'amv': ('Movies.csv', 'Video.csv'),
    'abe': ('Beauty.csv', 'Electronics.csv'),
    'afk': ('Food.csv', 'Kitchen.csv'),
    'amb': ('Movies.csv', 'Books.csv'),
    'afo_23': ('Food.csv', 'Office.csv'),
    'abh_23': ('Beauty_and_PC.csv', 'Health_and_PC.csv'),
    'abe_23': ('Beauty_and_PC.csv', 'Electronics.csv'),
    'amb_23': ('Movies.csv', 'Books.csv'),
    'abh_test': ('Beauty.csv', 'Health.csv'),
}

__all__ = [
    "read_raw",
    "txt_to_inter",
    "txt_to_jsons",
    "save_as_txt_ui",
    "save_as_txt_utsi",
    "save_as_inter",
    "save_as_jsons",
    "filter_non_overlapped",
    "filter_cold_item",
    "trim_sequence",
    "filter_mono_domain_user",
    "reindex_consistent",
    "reindex_independent",
    "count_item_in_TAT",
    "count_ui_txt",
    "reindex_item_from_processed_txt",
    "reindex_user_from_ui_txt",
    "reindex_item_from_ui_txt",

    "IDX_PAD",
    "MAPPING_FILE_NAME",
]