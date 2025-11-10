"""
    You can also run this script on terminal/with config
"""
from recbole.quick_start import run_recbole
from utils.config import config_sr

if __name__ == "__main__":

    run_recbole(model='FEARec', dataset='amv_video', config_dict=config_sr)

    # don't run this
    # run_recbole_cdr(model='CoNet', config_dict=config_cdr)
