import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse


def load_data(file):
    data = pd.read_csv(file)
    # test


if __name__ == "__main__":

    # List to store DataFrames from all .csv files
    input_file = 'final_dataset.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")
    args = parser.parse_args()

    input_file = args.file_input