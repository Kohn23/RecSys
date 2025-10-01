"""
    You can also run this script on terminal/with config
"""
from recbole_cdr.quick_start import run_recbole_cdr
from recbole.quick_start import run_recbole
from config.config_dicts import config_sr, config_cdr
from recbole.model.sequential_recommender import BERT4Rec, FEARec
from recbole_cdr.model.cross_domain_recommender.conet import CoNet


if __name__ == "__main__":

    run_recbole(model='FEARec', dataset='amv_video', config_dict=config_sr)

    # don't run this
    # run_recbole_cdr(model='CoNet', config_dict=config_cdr)
