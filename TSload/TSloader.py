"""Tsloader module."""

import pandas as pd
import numpy as np
import os
import shutil
import logging


class TSloader:
    """Use to write, load and modify timeseries dataset.

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

    Performance :
        It presupposes categories and 'number of features << number of feature
        entry'.

    Args:
        path (str, optional): The path to the dataset. datatype (str): The type
            of data which inform about the structure of the data. It is used as
            part of the file name in the dataset.
        datatype (str, optional): The type of data which inform about the
            structure of the data. It is used as part of the file name in the
            dataset.
        split (list[int] , optional): The sequence of splits to store on disk.
        subsplit_indices (list[int] , optional): The indices to use in subsplit.
            Default is to use all the indices from the split.
        subsplit_names (list[str] , optional): The subsplit scheme to use.
            Default is to use the whole split.
        parallel (bool, optional): Parallel informn on how to manipulate
            metadata. Parallel must be set to True to use in parallel to be used
            in parallel.
        permission (bool, optional): To choose between {'read', 'write',
            'overwerite'}, with an incresing level of permission for the loader.
            With 'write', you can only add data. Any operation that would remove
            data or metadata will raise an error. With 'read' you can read the
            data but not change it. With 'overwrite' you can do all operations.
            Default is 'write'

    Attributes:
        path (str): The path to the dataset.
        datatype (str, optional): The type of data which inform about the
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
        permission (str): Default to 'write'. The options are :

            - 'read' you can read the data but not change it.
            - 'write', you can only add data. Any operation that would remove data or
                metadata will raise an error.
            - 'overwrite' you can do all operations.

    """
    def __init__(
        self,
        path: str = "data",
        datatype: str = None,
        split: list[str] = None,
        subsplit_indices: list[int] = None,
        subsplit_names: list[str] = None,
        parallel: bool = False,
        permission: str = "write",
    ) -> "TSloader":
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
        self.create_path()

    def create_path(self) -> None:
        """Create the path if it doesn't exsist."""
        if not os.path.isdir(self.path):
            logging.info(f"Path '{self.path}' does not exist, creating.")
            os.makedirs(self.path)

    def append_path(self, filename: str) -> str:
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
            path (str):

        """
        if permission not in ["read", "write", "overwrite"]:
            raise ValueError("Permission is either 'read', 'write' or 'overwrite'")

        self.permission = permission

    def rm_dataset(self) -> None:
        """Remove dataset. Dangerous method.

        Raises:
            ValueError: If permission is not overwerite.

        """
        if self.permission != "overwrite":
            raise ValueError("To remove the dataset, you need the overwrite permission")

        shutil.rmtree(self.path)

    def move_dataset(self, new_path: str) -> None:
        """Move dataset to another location.

        Args:
            new_path (str):

        Raises:
            ValueError: If permission is not ovwerwrite or `self.path` is
                equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """
        old_path = self.path

        if self.permission != "overwrite":
            raise ValueError("To move the dataset, you need the overwrite permission")
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
            new_path (str):

        Raises:
            ValueError: If `self.path` is equal to `new_path`.
            OSError: If `new_path` directory exsists.

        """
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

    @staticmethod
    def merge_dataset(loaders, merge_path: str, **merge_loader_args: any) -> "TSloader":
        """Merge dataset.

        The merge path needs to be distinct from the path of all loaders.

        Args:
            loaders
            merge_path (str):
            **merge_loader_args (any):

        Returns:
            "TSloader": TSloader instance with the pandas' metadata attribute merged.

        Raises:
            ValueError: If `merge_path` is one of `loaders` path.

        """
        if type(loaders) is not list:
            loaders = [loaders]

        merge_loader = TSloader(merge_path, **merge_loader_args)

        i = 0
        for loader in loaders:
            if loader.path == merge_path:
                raise ValueError(
                    "The merge path needs to be distinct "
                    + "from the path of all loaders."
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
        metadata_file = self.append_path("metadata.pqt")
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
            raise ValueError("This loader is read-only.")
        if self.parallel:
            metadata_file = self.get_filename("metadata-")
        else:
            metadata_file = self.append_path("metadata.pqt")
        self.metadata.to_parquet(metadata_file)

    def add_datatype_to_metadata(self) -> None:
        """Add the current datatype to the metadata indices."""
        if self.metadata.empty:
            self.metadata = pd.DataFrame({"datatype": [self.datatype]})
            self.metadata.set_index(["datatype"], inplace=True)
        elif self.datatype not in self.metadata.index:
            datatype = self.metadata.index.append(pd.Index([self.datatype]))
            self.metadata = self.metadata.reindex(datatype, fill_value=[])
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

    def overwrite_metadata(self, **metadata: list[str]) -> None:
        """Overwrite metadata.

        Args:
            **metadata (list[str]):

        Raises:
            ValueError: If permission is not overwrite.

        """
        if self.permission != "overwrite":
            raise ValueError("This method needs the overwrite permission.")
        for key in metadata:
            if key not in self.metadata.columns:
                self.metadata[key] = ""
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
                self.overwrite_metadata(split=list(set(self.split)))
            else:
                raise ValueError(
                    "A split already exsists. "
                    + "To force this split, you need the overwrite permission."
                )
        else:
            # add the split to metadata
            self.add_metadata(split=split)

    def merge_metadata(self, write: bool = True, rm: bool = True) -> None:
        """Merge metadata without shared datatype between 'metadata-' file.

        A preprocess work must be done if datatype are shared to then
        use this method.

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
        if (self.permission == "read" and write):
            raise ValueError(
                "You cannot write metadata while merging "
                + "with 'read' permission."
            )
        elif self.permission != "overwrite" and rm:
            raise ValueError(
                "You cannot remove metadata while merging "
                + "without overwrite permission."
            )
        elif self.parallel:
            raise ValueError(
                "Set the parallel execution attribute " + "to `False` before merging."
            )

        elif not self.metadata.empty and self.permission != "overwrite":
            raise ValueError(
                "Trying to merge metadata but it already exists. "
                + "To force it, change the overwrite permission."
            )

        self.metadata = pd.DataFrame()
        for filename in os.listdir(self.path):
            if filename[0:9] == "metadata-":
                metadata_file = self.append_path(filename)
                new_metadata = pd.read_parquet(metadata_file)
                for datatype in new_metadata.index:
                    self.datatype = datatype
                    features = new_metadata["features"][0]
                    IDs = new_metadata["IDs"][0]
                    split = new_metadata["split"][0]

                    self.add_datatype_to_metadata()
                    self.add_metadata(split=split, IDs=IDs, features=features)

                # remove metadata-* file
                if rm:
                    os.remove(metadata_file)

        if write:
            self.write_metadata()

    #######################
    # datatype operations #
    #######################

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

        """
        self.datatype = datatype
        self.add_datatype_to_metadata()

        # Initialize and set the split
        self._initialize_split_metadata(split)
        self.set_split(subsplit_indices, subsplit_names)
        self.split_index = 0  # start at the beginning

    def set_split(
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
        return self.append_path(filename)

    def load(self) -> pd.DataFrame:
        """Load datatatype's data.

        Returns:
            pd.DataFrame: The pandas' data.

        """
        if self.datatype is None or not os.path.isfile(self.get_filename()):
            self.df = pd.DataFrame()
        else:
            self.df = pd.read_parquet(self.get_filename())
        return self.df

    def write(self) -> None:
        """Write datatatype's data.

        Raises:
            ValueError: If permission is only 'read' or if attribute `datatype` is not
                defined.

        """
        if self.permission == "read":
            raise ValueError("This loader permission is read-only.")
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
                "To initialize a non-empty datatype, you need the overwrite permission."
            )

        self.df = df.set_index(["ID", "timestamp"])

        self.overwrite_metadata(IDs=list(self.df.index.droplevel(1).unique()))
        self.overwrite_metadata(features=list(self.df.columns.unique()))

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
            raise ValueError("To remove a datatype, you need the overwrite permission")
        elif self.df.empty():
            raise ValueError("Trying to remove not existing datatype.")
        self.df = pd.DataFrame()

        if rm_from_metadata:
            self.metadata.drop(self.datatype, inplace=True)

    #######################
    # add data to dataype #
    #######################

    def add_ID(
        self, df: pd.DataFrame = None, ID: str = None, append: bool = True
    ) -> None:
        """Add ID to datatype.

        Args:
            df (pd.DataFrame): A dataframe with data for a given `ID`.
            ID (str): The unique identication name for the data.
            append (bool, optional): Whether or not to append the `df` to an
                existing ID in `self.df`. Default is `True`

        Raises:
            ValueError: If `ID` or `df` are not well-defined or if trying to
                overwrite data without the permisison.

        """
        if df is None or "timestamp" not in df.columns:
            raise ValueError("Need a well-defined DataFrame.")
        elif ID is None:
            raise ValueError("Need an ID.")

        if ID in self.df.index:
            if self.permission == "overwrite":
                self.rm_ID(
                    ID, rm_from_metadata=False
                )  # Metadata needs to be kept the same
                # self.df.drop(ID, level=0, inplace=True)
            elif not append:
                raise ValueError(
                    f"{ID} already in DataFrame. "
                    + "Append to it by using the paramater of the method, "
                    + "or overwrite it by changing the permission."
                )

        df["ID"] = ID
        df.set_index(["ID", "timestamp"], inplace=True)
        self.df = pd.concat([self.df, df], axis=0)

        self.add_metadata(IDs=ID, features=self.df.columns)  # add to metadata

    def add_feature(
        self, df: pd.DataFrame = None, ID: str = None, feature: str = None
    ) -> None:
        """Add feature to ID in datatype.

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

        if ID not in self.df.index:
            # ID not in self.df, use the `add_ID` method
            self.add_ID(df, ID)  # Metadata handled there
        elif feature in self.df.columns:
            # ID and feature are in self.df
            if self.permission == "overwrite":
                # You need to overwrite the ID, to have same input length
                feature_df = pd.DataFrame(df[["timestamp", feature]]).set_index(
                    ["timestamp"], drop=True
                )
                # join features, keep the newest
                df_ID = (
                    self.df.loc[ID]
                    .join(feature_df, how="right", lsuffix="drop")
                    .drop(feature + "drop", axis=1)
                    .reset_index()
                )

                # You need to overwrite the ID, to have same input length
                self.add_ID(df_ID, ID)  # Metadata handled there

            else:
                raise ValueError(
                    f"{feature} already in DataFrame. "
                    + "To force it, you can change the overwrite permission."
                )
        else:
            # ID is in self.df but feature is not in self.df
            # Overwrite ID row
            feature_df = pd.DataFrame(df[["timestamp", feature]]).set_index(
                ["timestamp"], drop=True
            )
            # join features to re-create the DatFrame for ID
            df_ID = self.df.loc[ID].join(feature_df, how="outer").reset_index()
            # You need to overwrite the ID, to have same input length
            self.add_ID(df_ID, ID)  # Metadata handled there

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
            raise ValueError("To remove an ID, you need the overwrite permission.")

        elif ID not in self.df.index:
            raise ValueError("ID does not exsit and trying to remove it.")

        # update df
        self.df.drop(ID, level=0, inplace=True)
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
            raise ValueError("To remove a feature, you need the overwrite permission")

        elif feature not in self.df.columns:
            raise ValueError("Trying to remove not existing feature.")

        # update df
        self.df.drop(feature, axis=1, inplace=True)
        if rm_from_metadata:
            self.overwrite_metadata(features=list(self.df.columns.unique()))
