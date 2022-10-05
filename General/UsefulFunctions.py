import operator
import os

from typing import Tuple

import numpy as np
import pandas as pd


def find_files_in_folder(folder, filter="any", inc="") -> list:
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    if filter == "any":
        pass
    elif filter == "dir":
        files = [file for file in files if os.path.isdir(file)]
    elif filter == "file":
        files = [file for file in files if not os.path.isdir(file)]
    elif filter == "include":
        if inc == "":
            print("No include filter given")
        else:
            files = [file for file in files if inc in os.path.basename(file)]
    else:
        print("Filter type is invalid")
    return files


def index_df_list(df, df_list):
    for index, df_list in enumerate(df_list):
        try:
            if (df_list.values == df.values).all():
                return index
        except AttributeError:
            print("dataframe not found in dataframe list")


def moving_average(data, size) -> list:
    data = list(data)
    return np.convolve(data, np.ones(size), "valid") / size


def get_parent_path(path) -> str:
    return os.path.abspath(os.path.join(path, ".."))


def get_file_name(path, ext="") -> str or None:
    file_name = os.path.basename(path)
    if not ext:
        return file_name
    else:
        return file_name.replace(ext, "")


def get_max_index(data) -> Tuple[int, float]:
    max_index, max_value = max(enumerate(data), key=operator.itemgetter(1))

    return max_index, max_value


def clear_df(df) -> pd.DataFrame:
    return df.iloc[0:0]
