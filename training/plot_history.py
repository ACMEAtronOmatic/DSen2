#!/usr/bin/env python3

import os ; import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from argparse import ArgumentParser
import yaml
import gc

plt.style.use('ggplot')

DESCRIPTION = 'Plotting history loss from Pan-sharpening CNN model.'

def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('pandas_file', help = 'Location pandas DataFrame file.')

    args = parser.parse_args()
    pandasFile = args.pandas_file

    for opener in [pd.read_pickle, pd.read_csv, pd.read_json, pd.read_hdf]:
        try:
            df = opener(pandasFile)
            break
        except:
            pass

    print(df)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot()

    df['loss'].plot(ax=ax, marker='o', mec='k')
    df['val_loss'].plot(ax = ax, marker='o', mec='k')

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.savefig('history.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
