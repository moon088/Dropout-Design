from __future__ import print_function

import csv
import glob
import itertools
import os
import re

def remove(path):
    for f in glob.glob(path):
        print(''.join(['removed ', str(f)]))
        os.remove(f)

def remove_csvs():
    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    models_path = os.path.abspath(src_path + '/models')
    # remove all csv files
    remove(models_path + '/*.csv')

if __name__ == '__main__':
    remove_csvs()
