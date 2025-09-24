"""
    You can also run this script on terminal/with config
"""
from recbole_cdr.quick_start import run_recbole_cdr
from recbole.quick_start import run_recbole
from config.config_dicts import config_sr, config_cdr
from recbole.model.sequential_recommender import SASRec, FEARec
from recbole_da.model.sequential_recommender import CL4SRec

if __name__ == "__main__":

    run_recbole(model='BERT4Rec', dataset='abh_beauty', config_dict=config_sr)
