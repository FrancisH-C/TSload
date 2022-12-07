"""DataFormat module.

These are example of how to format data.

"""
from TSload.TSloader import TSloader
import os
import pandas as pd
from typing import Callable
import shutil
import multiprocessing


class LoadersProcess(multiprocessing.Process):
    """A collection of loaders and a function to apply to them using multiprocessing.

    Args:
        loaders (TSLoader): Sets attribute of the same name.
        function Callable[[TSloader], None]): Sets attribute of the same name.

    Attributes:
        loaders ("TSLoader"): List of loaders to use with their split for
            multiprocessing.
        function Callable[["TSloader"], None]): A function to apply to every split of
            every loaders.

    """

    def __init__(self, loaders: "TSloader", function: Callable[["TSloader"], None]):
        """Init method."""
        super(LoadersProcess, self).__init__()
        self.loaders = loaders
        self.function = function

    def run(self):
        """Multiprocessing run function.

        For every loaders, load their split and apply the function from attribute
        `function`.

        """
        for loader in self.loaders:
            loader.reset_split_index()
            for split in loader.split:
                self.function(loader)
                loader.next_split_index()


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


def merge_dataset(
    loaders: "TSloader", merge_path: str, **merge_loader_args: any
) -> "TSloader":
    """Merge dataset assuming no shared dataype.

    The merge path needs to be distinct from the path of all loaders.

    Args:
        loaders (TSloader): List of loaders to merge data on.
        merge_path (str): List of loaders to merge data on.
        **merge_loader_args (any): Arguments for the outputed TSloader's constructor

    Returns:
        "TSloader": TSloader instance with the metadata attribute merged.

    Raises:
        ValueError: If `merge_path` is one of `loaders` path.

    """
    if type(loaders) is not list:
        raise ValueError("Give a list of the loaders to merge")

    merge_loader = TSloader(
        merge_path, loaders[0].datatype, permission="overwrite", **merge_loader_args
    )

    i = 0
    for loader in loaders:
        if loader.path == merge_path:
            raise ValueError(
                "The merge path needs to be distinct " + "from the path of all loaders."
            )
        for filename in os.listdir(loader.path):
            if filename == "metadata.pqt":
                src = os.path.join(loader.path, filename)
                dst = os.path.join(merge_path, "metadata-" + str(i) + ".pqt")
                shutil.copyfile(src, dst)
                i += 1
            else:
                src = os.path.join(loader.path, filename)
                dst = os.path.join(merge_path, filename)
                shutil.copyfile(src, dst)

    merge_loader.merge_metadata(write=True, rm=True)
    return merge_loader
