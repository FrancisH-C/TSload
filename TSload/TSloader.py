"""Tsloader module."""
import pandas as pd
import numpy as np
import os
import shutil
import logging
import multiprocessing
from typing import Callable
from TSload.DataFormat import np_to_TSdf, dict_to_TSdf, df_to_TSdf


class TSloader:
    """Use to write, load and modify a timeseries dataset.

    UPDATE : Create a class datatype. This would simplify TSloader and its uses

    A TSloader is assigned a path to a "dataset". Optionally, it can have a
    "datatype" which informs about the structure of the data. "Datatype" is a
    collection of multiple input with different "IDs". A given "datatype" as the
    exact same "features" which is the data indexed with a "timestamp" (the
    timeseries).

    A "datatype" can be splitted on different files on disk, this is called a
    "split". A TSloader with that "datatype" and a "subsplit" (either with names
    or indices) can manipulate the data from the files. It is used when a single
    datatype is too large or for parallelization purposes.

    Notes :
        Most of the attributes are better changed using their 'set' method or by
        pdefining a new loader.

    Args:
        path (str): Sets attribute of the same name.
        datatype (str): Sets attribute of the same name.
        split (list[str], optional): Sets attribute of the same name.
        subsplit_indices (list[int] , optional): The indices to use in subsplit.
            Default is to use all the indices from the split.
        subsplit_names (list[str] , optional): The subsplit scheme to use.
            Default is to use the whole split.
        parallel (bool, optional): Sets attribute of the same name.
        permission (str, optional): Sets attribute of the same name.

    Attributes:
        path (str): The path to the dataset.
        datatype (str): The type of data which inform about the
            structure of the data. It is used as part of the file name in the
            dataset.
        df (pd.DataFrame): The pandas' dataset.
        metadata (pd.DataFrame): The pandas' metadata.
        split (list[str], optional): A given datatype is store in a sequence of
            splits. Used when a single datatype is too large or for
            parallelization.
        parallel (bool): Parallel informn on how to manipulate metadata.
            Parallel must be set to True to use in parallel to be used in
            parallel. Default is False.
        permission (str): To choose between {'read', 'write', 'overwerite'},
            with an increasing level of permission for the loader. The options are :

            - 'read' : Read only the data on disk and change it on memory.
            - 'write' : Add only new datatype and new ID. Any operation that \
               would remove or change data or metadata will raise an error.
            - 'overwrite' (default) :  Do all operations and write on disk.

    """

    def __init__(
        self,
        path: str = "data",
        datatype: str = None,
        split: list[str] = None,
        subsplit_indices: list[int] = None,
        subsplit_names: list[str] = None,
        parallel: bool = False,
        permission: str = "overwrite",
    ) -> "TSloader":
        """Init method."""
        # Permissions
        self.set_permission(permission)  # read, write, write

        # For parallel usage
        self.parallel = parallel

        # Select a dataset and load its metadata
        self.set_path(path)
        self.load_metadata()

        # Set the datatype and use it to load datatype's data.
        self.set_datatype(datatype, split, subsplit_indices, subsplit_names)
        self.df = self.load()

    ######################
    # dataset operations #
    ######################

    def set_path(self, path: str) -> None:
        """Set the current path.

        Args:
            path (str): The path to set.

        """
        self.path = path
        self._create_path()

    def _create_path(self) -> None:
        """Create the path if it doesn't exsist."""
        if not os.path.isdir(self.path):
            logging.info(f"Path '{self.path}' does not exist, creating.")
            os.makedirs(self.path)

    def _append_path(self, filename: str) -> str:
        """Give the filename appended with the path attribute.

        Args:
            filename (str): Name of the file.

        Returns:
            str: Filename with appended the loader path.

        """
        return os.path.join(self.path, filename)

    def set_permission(self, permission="write") -> None:
        """Set the current path.

        Args:
            permission (str, optional): To choose between {'read', 'write',
                'overwerite'}, with an increasing level of permission for the loader.
                The options are :

                - 'read' : Read only the data on disk and change it on memory.
                - 'write' : Add only new datatype and new ID. Any operation that \
                   would remove or change data or metadata will raise an error.
                - 'overwrite' (default) :  Do all operations and write on disk.

        """
        if permission not in ["read", "write", "overwrite"]:
            raise ValueError("Permission is either 'read', 'write' or 'overwrite'")

        self.permission = permission

    def rm_dataset(self, ignore_errors=True) -> None:
        """Remove dataset. Dangerous method.

        Args:
            ignore_errors (str, optional): If ignore_errors is set to True, errors
                    arising due to unsuccessful file removals will be ignored. This
                    is set to `True` by default

        Raises:
            ValueError: If permission is not overwerite.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove the dataset, you need 'overwrite' permission")

        shutil.rmtree(self.path, ignore_errors=ignore_errors)

    def move_dataset(self, new_path: str) -> None:
        """Move dataset to another location.

        Args:
            new_path (str): The path where to move the data.

        Raises:
            ValueError: If permission is not ovwerwrite or `self.path` is
                equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """
        old_path = self.path

        if self.permission != "overwrite":
            raise ValueError("To move the dataset, you need 'overwrite' permission")
        elif old_path == new_path:
            raise ValueError(f"'{new_path}' is already the current dataset path.")
        try:
            shutil.move(old_path, new_path)
            self.set_path(new_path)
        except OSError:
            raise OSError(
                f"'{new_path}' already exists, "
                + "to merge dataset use `merge_dataset`."
            )

    def copy_dataset(self, new_path: str) -> None:
        """Copy dataset to another location.

        Args:
            new_path (str): The path where to copy the data.

        Raises:
            ValueError: If `self.path` is equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """
        if self.permission == "read":
            raise ValueError("To copy the dataset, you need 'write' permission")

        old_path = self.path

        if old_path == new_path:
            raise ValueError(f"'{new_path}' is already the current dataset path.")
        try:
            shutil.copytree(old_path, new_path)
            self.set_path(new_path)
        except OSError:
            raise OSError(
                f"'{new_path}' already exists, "
                + "to merge dataset use `merge_dataset`."
            )

    #######################
    # metadata operations #
    #######################

    def toggle_parallel(self) -> None:
        """Toggle parallel option."""
        if self.parallel:
            self.parallel = False
            print("Parallel mode deactivated")
        else:
            self.parallel = True
            print("Parallel mode activated")

    def load_metadata(self) -> pd.DataFrame:
        """Load datataset's metadata.

        Returns:
            pd.DataFrame: The pandas' metadata.

        """
        metadata_file = self._append_path("metadata.pqt")
        if os.path.isfile(metadata_file):
            self.metadata = pd.read_parquet(metadata_file)
        else:
            self.metadata = pd.DataFrame()

        return self.metadata

    def write_metadata(self) -> None:
        """Write datataset's metadata.

        Raises:
            ValueError: If permission is read.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        if self.parallel:
            metadata_file = self.get_filename("metadata-")
        else:
            metadata_file = self._append_path("metadata.pqt")
        self.metadata.to_parquet(metadata_file)

    def _add_datatype_to_metadata(self) -> None:
        """Add the current datatype to the metadata indices."""
        if self.metadata.empty:
            self.metadata = pd.DataFrame({"datatype": [self.datatype]})
            self.metadata.set_index(["datatype"], inplace=True, drop=False)
            self.metadata["IDs"] = [[]]
            self.metadata["features"] = [[]]
        elif self.datatype not in self.metadata.index:
            datatype = self.metadata.index.append(pd.Index([self.datatype]))
            self.metadata = self.metadata.reindex(datatype, fill_value=[])
            self.metadata["datatype"] = datatype
        # else datatype is already in metadata indices

    def add_metadata(self, **metadata: list[str]) -> None:
        """Verify if entry is already there before append.

        Args:
            **metadata (list[str]):

        """
        for key in metadata:
            if key not in self.metadata.columns:
                self.metadata[key] = ""
                self.metadata.at[self.datatype, key] = metadata[key]

            updated_metadata = list(
                set(np.append(self.metadata.at[self.datatype, key], [metadata[key]]))
            )
            self.metadata.at[self.datatype, key] = updated_metadata

    def update_metadata(self) -> None:
        """Verify if entry is already there before append."""
        IDs = list(set(self.df["ID"]))
        features = list(self.df.columns.drop(["ID", "timestamp", "dim"]))

        self.metadata.at[self.datatype, "IDs"] = IDs
        self.metadata.at[self.datatype, "features"] = features

    def overwrite_metadata(self, **metadata: list[str]) -> None:
        """Overwrite metadata.

        Args:
            **metadata (list[str]):

        Raises:
            ValueError: If permission is not overwrite.

        """
        if self.permission != "overwrite":
            raise ValueError("This method needs 'overwrite' permission.")
        for key in metadata:
            if key not in self.metadata.columns:
                self.metadata[key] = ""
            if type(metadata[key]) is not list:
                metadata[key] = [metadata[key]]
            self.metadata.at[self.datatype, key] = metadata[key]

    def _initialize_split_metadata(self, split: list[str] = None) -> None:
        """Initialize split metadata.

        Args:
            split (list[str], optional):

        Raises:
            ValueError: If split exsists and no overwrite permission is granted.

        """
        if split is None:
            self.add_metadata(split=[])
        # If split is in metadata
        elif self.datatype in self.metadata.index and "split" in self.metadata.columns:
            if self.permission == "overwrite":
                self.split = split
                # overwrite the split to metadata
                self.overwrite_metadata(split=list(self.split))
            else:
                raise ValueError(
                    "A split already exsists. "
                    + "To force this split, you need the overwrite permission."
                )
        else:
            # add the split to metadata
            self.add_metadata(split=split)

    def merge_metadata(self, write: bool = True, rm: bool = True) -> None:
        """Merge metadata between 'metadata-' file.

        Args:
            write (bool, optional): Whether or not to write the merged metadata.
            rm (bool, optional): Whether or not to removed all the 'metadata-' files
                after the merge.

        Raises:
            ValueError: If trying to write metadata with 'read'
                permission; Or if trying to remove metadata (on disk or
                memory) without 'overwrite' permission; Or if `parallel`
                attribute is `True`.

        """
        if self.permission == "read" and write:
            raise ValueError(
                "You cannot write metadata while merging " + "with 'read' permission."
            )
        elif self.permission != "overwrite" and rm:
            raise ValueError(
                "You cannot remove metadata while merging "
                + "without 'overwrite' permission."
            )

        elif not self.metadata.empty and self.permission != "overwrite":
            raise ValueError(
                "Trying to merge metadata but it already exists. "
                + "To force it, change permission to 'overwrite'."
            )

        if self.parallel:
            raise ValueError(
                "Set the parallel execution attribute " + "to `False` before merging."
            )

        self.metadata = pd.DataFrame()
        for filename in os.listdir(self.path):
            if filename[0:9] == "metadata-":
                metadata_file = self._append_path(filename)
                new_metadata = pd.read_parquet(metadata_file)
                for datatype in new_metadata.index:
                    self.datatype = datatype
                    features = new_metadata["features"][0]
                    IDs = new_metadata["IDs"][0]
                    split = new_metadata["split"][0]

                    self._add_datatype_to_metadata()
                    self.add_metadata(split=split, IDs=IDs, features=features)

                # remove metadata-* file
                if rm:
                    os.remove(metadata_file)

        if write:
            self.write_metadata()

    #######################
    # datatype operations #
    #######################

    def get_df(self, IDs=None, timestamps=None, dims=None, drop=True):
        """Get DataFrame for the datetype.

        If "IDs", "timestamps" or "dims" is specify, fix that value. Otherwise
        returns the entries corrresponding to all values. The more efficient in
        memory is to have, at most, one specific fix value.

        """
        df = pd.DataFrame.copy(self.df)
        if df.empty:
            return df

        # Return entries for fixed value or all values.
        if IDs is not None:
            if timestamps is not None:
                if dims is not None:
                    df = df.loc[IDs, timestamps, dims]
                else:  # dims is None
                    df.set_index(["ID"], inplace=True, drop=False)
                    df = (
                        df.loc[IDs].set_index(["timestamp"], drop=False).loc[timestamps]
                    )
            else:  # timestamps is None
                if dims is not None:
                    df.set_index(["ID"], inplace=True, drop=False)
                    df = df.loc[IDs].set_index(["dim"], drop=False).loc[dims]
                else:  # dims is None
                    df = df.loc[IDs]
        else:  # IDs is None
            if timestamps is not None:
                if dims is not None:
                    df.set_index(["timestamp"], inplace=True, drop=False)
                    df = df.loc[timestamps].set_index(["dim"], drop=False).loc[dims]
                else:  # dims is None
                    df.set_index(["timestamp"], inplace=True, drop=False)
                    df = df.loc[timestamps]

            df.set_index(["ID", "timestamp", "dim"], inplace=True, drop=drop)
        return df

    def set_datatype(
        self,
        datatype: str,
        split: list[str] = None,
        subsplit_indices: list[int] = None,
        subsplit_names: list[str] = None,
    ) -> None:
        """Change datatype and split used to load data.

        Args:
            datatype (str): The datatype to set.
            split (list[str], optional): The split to set.
            subsplit_indices (list[int], optional): The split indices to set.
            subsplit_names (list[str], optional): The split names to set.

        Raises:
            ValueError: If `datatype` is undefined.

        """
        if datatype is None:
            raise ValueError("You need to define a datatype")

        self.datatype = datatype
        self._add_datatype_to_metadata()

        # Initialize and set the split
        self._initialize_split_metadata(split)
        self._set_split(subsplit_indices, subsplit_names)
        self.split_index = 0  # start at the beginning

    def _set_split(
        self, subsplit_indices: list[int] = None, subsplit_names: list[str] = None
    ) -> None:
        """Set split.

        Args:
            subsplit_indices (list[int], optional): The split indices to set.
            subsplit_names (list[str], optional): The split names to set.

        Raises:
            ValueError: If both `subsplit_indices` and `subsplit_names`
                are given as parameters or if they are, respectively,
                invalid for the data's split.

        """
        if subsplit_indices is not None and subsplit_names is not None:
            raise ValueError(
                "Give either subsplit_indices or subsplit_names, not both."
            )
        elif subsplit_indices is None and subsplit_names is None:
            self.split = self.metadata.at[self.datatype, "split"]
        elif subsplit_indices is not None:
            split = self.metadata.at[self.datatype, "split"]
            if max(subsplit_indices) < len(split):
                self.split = [split[i] for i in subsplit_indices]
            else:
                raise ValueError("Invalid split indices.")
        else:  # subsplit_names is not None:
            split = self.metadata.at[self.datatype, "split"]
            if set(subsplit_names).issubset(split):
                self.split = subsplit_names
            else:
                raise ValueError("Invalid split names.")

    def reset_split_index(self) -> None:
        """Reset split index to 0."""
        self.split_index = 0

    def next_split_index(self) -> None:
        """Increment split index by 1."""
        self.split_index += 1

    def set_split_index(self, index: int) -> None:
        """Set the split index.

        Args:
            index (int): Value to set the current split index.

        """
        self.split_index = index

    def get_filename(self, prefix: str = "") -> str:
        """Get the filename to load for current datatype and split_index.

        Args:
            prefix (str, optional): A prefix to add to a filename.

        Returns:
            str: The filename to load for current datatype and split_index.

        """
        if len(self.split) > 0:
            filename = (
                prefix + self.datatype + "-" + self.split[self.split_index] + ".pqt"
            )
        else:
            filename = prefix + self.datatype + ".pqt"
        return self._append_path(filename)

    def load(self) -> pd.DataFrame:
        """Load datatatype's data.

        Returns:
            pd.DataFrame: The pandas' data.

        """
        filename = self.get_filename()
        if self.datatype is None or not os.path.isfile(filename):
            self.df = pd.DataFrame()
        else:
            self.df = pd.read_parquet(filename)
        return self.df

    def write(self) -> None:
        """Write datatatype's data.

        Raises:
            ValueError: If permission is only 'read' or if attribute `datatype` is not
                defined.

        """
        if self.permission == "read":
            raise ValueError("This loader has only 'read' permission.")
        elif self.datatype is None:
            raise ValueError("No defined datatype.")

        self.df.to_parquet(self.get_filename())
        self.write_metadata()

    def initialize_datatype(self, df: pd.DataFrame = None) -> None:
        """Initialize datatatype's with data.

        Args:
            df (pd.DataFrame): A dataframe with data for the datatype.

        Raises:
            ValueError: If trying to overwrite data without 'overwrite' permission
                or `df` is not well-defined.

        """
        if df is None or "ID" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("Need a well-defined DataFrame.")
        elif len(self.df) > 0 and self.permission != "overwrite":
            raise ValueError(
                "To initialize a non-empty datatype, you need 'overwrite' permission."
            )

        self.df = df_to_TSdf(df)

        #self.update_metadata()

    def rm_datatype(self, rm_from_metadata: bool = True) -> None:
        """Remove datatatype's data.

        Args:
            rm_from_metadata (bool, optional): If the datatype should also be removed
                from metadata. Default is True.

        Raises:
            ValueError: If permission is not overwrite or `self.path` is equal to
                `new_path`.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove a datatype, you need 'overwrite' permission")
        elif self.df.empty:
            raise ValueError("Trying to remove not existing datatype.")
        self.df = pd.DataFrame()

        if rm_from_metadata:
            self.metadata.drop(self.datatype, inplace=True)

    #######################
    # add data to dataype #
    #######################

    def add_ID(
        self,
        data=None,
        ID: str = None,
        dim_label: np.ndarray = None,
        timestamp: np.ndarray = None,
        collision: str = "overwrite",
    ) -> None:
        """Add ID to datatype.

        Caution, it Changes df. `df`'s columns could include "ID", "timestamp",
        "dim". If they don't have either, one will be provided for them.



        If no dim_label is given, assumes the number of dependent dimension is 1.

        Args:
            df (pd.DataFrame): A dataframe with data for a given `ID`.
            ID (str): The unique identication name for the data.
            collision (str, optional): To choose between {'ignore', 'append',
                'update', 'overwerite'}

                - 'overwrite' (default) : Overwrite the value.
                - 'update' : Updates the value.
                - 'ignore' : Does nothing.
                - 'append' : Append without index verification df
                   Dangerous: could lead to multiple timestamp problem.

        Raises:
            ValueError: If `ID` is not well-defined or if trying to
                overwrite data without the permisison.

        """
        # format the data depending of the type
        if type(data) is pd.DataFrame:
            df = df_to_TSdf(data, ID=ID, timestamp=timestamp, dim_label=dim_label)
        elif type(data) is np.ndarray:
            df = np_to_TSdf(data, ID=ID, timestamp=timestamp, dim_label=dim_label)
        elif type(data) is dict:
            df = dict_to_TSdf(data, ID=ID, timestamp=timestamp, dim_label=dim_label)
        else:
            raise ValueError("Data is of the wrong type")

        # ID
        if ID is None:
            if "ID" in df.columns:
                ID = df["ID"][0]
            else:
                raise ValueError("Need an ID.")

        if ID in self.df.index:
            if collision == "ignore":
                return
            elif collision == "append":
                # append all the info for the ID
                df = pd.concat([self.df.loc[ID], df], axis=0)

                # remove previous now duplicated data
                self.df.drop(index=ID, level="ID", inplace=True)
                self.df = pd.concat([self.df, df], axis=0)

            elif collision == "update" and self.permission == "overwrite":
                self.df = df.combine_first(self.df)
            elif collision == "overwrite" and self.permission == "overwrite":
                self.rm_ID(ID, rm_from_metadata=False)  # Keep metadata
                self.df = pd.concat([self.df, df], axis=0)
            else:
                raise ValueError(
                    "Trying to 'overwrite' an ID without permission; "
                    " Or collision parameter not valid"
                )
        else:
            # Append the ID to `self.df`.
            self.df = pd.concat([self.df, df], axis=0)
            self.update_metadata()  # add to metadata

    def add_feature(
        self, df: pd.DataFrame = None, ID: str = None, feature: str = None
    ) -> None:
        """Add feature to ID in datatype, merging on 'timestamp'.

        This method needs the overwrite permission because you overwrite an ID by
        providing a feature, hence changing the length of the ID.

        If ID is not specify, add it to all datatype. If feature already present and
        not overwrite, gives a warning.  To use `add_feature`, you need overwrite
        permission, because you overwrite the previous features to have the same
        lenght as the added `feature`.

        Args:
            df (pd.DataFrame): A dataframe with a `feature` column for a given `ID`.
            ID (str): The unique identication name for the data.
            feature (str): The feature name for the column.

        Raises:
            ValueError: If `ID`, `feature` or `df` are not well-defined or if trying to
                overwrite data without the permisison.

        """
        if df is None or "timestamp" not in df.columns or feature not in df.columns:
            raise ValueError("Need a well-defined DataFrame.")
        elif ID is None:
            raise ValueError("Need an ID.")
        elif feature is None:
            raise ValueError("Need a feature.")

        if self.permission != "overwrite":
            raise ValueError("Trying to `add_feature` without 'overwrite' permission")

        if ID not in self.df.index:
            # ID not in self.df, use the `add_ID` method
            self.add_ID(df, ID)  # Metadata handled there
        elif feature not in self.df.columns:
            # ID is in self.df and feature is not in self.df
            # Overwrite ID row
            current_ID = self.df.loc[ID].reset_index(drop=True)
            df_ID = df.combine_first(current_ID)
            self.add_ID(df_ID, ID, collision="overwrite")  # Metadata handled there
        else:
            # ID and feature are in self.df
            current_ID = self.df.loc[ID].reset_index(drop=True)
            df.reset_index(inplace=True, drop=True)
            df_ID = df.combine_first(current_ID)
            # You need to overwrite the ID, to have same input length
            self.add_ID(df_ID, ID, collision="overwrite")  # Metadata handled there

    ############################
    # remove data from dataype #
    ############################

    def rm_ID(self, ID: str = None, rm_from_metadata: bool = True) -> None:
        """Remove ID to datatype.

        Args:
            ID (str): The unique identication name for the data.
            rm_from_metadata (bool, optional): If the `ID` should also be removed from
                metadata. Default is True.

        Raises:
            ValueError: If permission is not "overwrite" or if `ID` is not in the
                index of `self.df`.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove an ID, you need 'overwrite' permission.")

        elif ID not in self.df.index:
            raise ValueError("ID does not exsit and trying to remove it.")

        # update df
        self.df.drop(index=ID, level="ID", inplace=True)
        if rm_from_metadata:
            self.overwrite_metadata(IDs=list(self.df.index.droplevel(1).unique()))

    def rm_feature(self, feature: str = None, rm_from_metadata: bool = True) -> None:
        """Remove feature to datatype.

        Args:
            feature (str): The feature name for the column.
            rm_from_metadata (bool, optional): If the `feature` should also be removed
                from metadata. Default is True.

        Raises:
            ValueError: If permission is not "overwrite" or if `feature` is not in the
                columns of `self.df`.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove a feature, you need 'overwrite' permission")
        elif feature not in self.df.columns:
            raise ValueError("Trying to remove not existing feature.")

        # update df
        self.df.drop(columns=feature, inplace=True)
        if rm_from_metadata:
            self.overwrite_metadata(features=list(self.df.columns.unique()))


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
