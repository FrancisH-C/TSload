"""DataFormat module.

These are example of how to format data.

"""
from TSload import TSloader
import os
import pandas as pd
from typing import Callable


def csv2pqt(
    path: str,
    filename: str,
    process_function: Callable[[pd.DataFrame], None] = None,
    **loader_args: any
) -> None:
    """Format a file from csv to pqt.

    Args:
        path (str): Path of the file.
        filename (str): Name of the file.
        process_function (Callable[[pd.DataFrame]) : None] ) : Default is  None.
            A function to processs the data
        **loader_args (any): Keyword arguments for the loader.

    """
    datatype, ext = os.path.splitext(filename)
    if ext != ".csv":
        raise ValueError("`Filename` should be a csv file")

    loader = TSloader(path, datatype, **loader_args)
    df = pd.read_csv(os.path.join(path, filename))

    # process the data
    if process_function is not None:
        process_function(df)

    loader.add_datatype(df)
    loader.write()


def dataset_csv2pqt(
    path: str,
    process_function: Callable[[pd.DataFrame], None] = None,
    **loader_args: any
) -> None:
    """Format files in path from csv to pqt.

    Args:
        path (str): Path of the file.
        filename (str): Name of the file.
        process_function (Callable[[pd.DataFrame]) : None] ) : Default is  None.
            A function to processs the data
        **loader_args (any): Keyword arguments for the loader.

    """
    for filename in os.listdir(path):
        _, ext = os.path.splitext(filename)
        if ext == ".csv":
            csv2pqt(path, filename, **loader_args)
