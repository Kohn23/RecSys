from utils.data.read import *
from utils.data.filtering import *
from utils.data.file_conversion import *
from utils.data.save import *
from utils.data.count import *
from utils.data.reindex import *

IDX_PAD = 0

MAPPING_FILE_NAME = {
    'afo': ('Food.csv', 'Office.csv'),
    'abh': ('Beauty.csv', 'Health.csv'),
    'amv': ('Movies.csv', 'Video.csv'),
    'abe': {'Beauty.csv', 'Electronics.csv'}
}

__all__ = [
    "read_two_domains",
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

    "IDX_PAD",
    "MAPPING_FILE_NAME",
]