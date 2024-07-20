import argparse
import glob
import itertools
import os

import prediction
import remove
import accumulation

def evaluate():
    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    models_path = os.path.abspath(src_path + '\\models')

    # model evaluation & prediction
    # remove all csv files
    # remove.remove_csv()
    # for h5 in glob.glob(models_path + '\\*'):
    #     prediction.evaluate_model(h5)
    accumulation.accumulate()

if __name__ == '__main__':
    evaluate()