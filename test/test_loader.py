import pandas as pd
import numpy as np
from TSload import TSloader, DataFormat
import pytest


def same_data(df1, df2):
    df1 = df1.fillna(-1)
    df2 = df2.fillna(-1)

    if df1.shape != df2.shape:
        return False

    for ID in df1.index:
        for name in df1.columns:
            if df1.loc[ID, name] != df2.loc[ID, name]:
                return False
    return True


def simple_loader():
    """Simple loader for test."""
    path = "data/test_add"
    datatype = "simulated"
    permission = "overwrite"
    loader = TSloader(path, datatype, permission=permission)

    ID = "added_ID"
    feature = "added_feature"

    d = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
    }
    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}
    df = pd.DataFrame(data=d)
    df_feature = pd.DataFrame(data=d_feature)

    loader.initialize_datatype(df=df)
    loader.add_ID(df, ID=ID, collision="overwrite")
    loader.add_feature(df_feature, ID=ID, feature=feature)

    return loader


def test_permission():
    testpath = "data/test_nonexistent"
    loader = simple_loader()

    loader.set_permission("read")
    # Dataset operations
    with pytest.raises(ValueError):
        loader.copy_dataset(testpath)
    with pytest.raises(ValueError):
        loader.write()
    with pytest.raises(ValueError):
        loader.write_metadata()
    with pytest.raises(ValueError):
        loader.merge_metadata()
    with pytest.raises(ValueError):
        loader.rm_dataset()
    with pytest.raises(ValueError):
        loader.move_dataset(testpath)

    # Overwrite operations
    loader.set_permission("write")
    ID = "added_ID"
    feature = "feature0"
    df = pd.DataFrame(columns=["ID", "timestamp", feature])

    with pytest.raises(ValueError):
        loader.rm_ID(ID)
    with pytest.raises(ValueError):
        loader.rm_feature(feature)

    with pytest.raises(ValueError):
        loader.initialize_datatype(pd.DataFrame(columns=["ID", "timestamp"]))

    # You can append but not overwrite
    loader.add_ID(df.copy(), ID=ID, collision="append")
    with pytest.raises(ValueError):
        loader.add_ID(df.copy(), ID=ID, collision="overwrite")
    with pytest.raises(ValueError):
        loader.add_feature(df.copy(), ID=ID, feature=feature)


def test_dataset_operations():
    loader = simple_loader()
    loader_other = simple_loader()

    loader.set_path("data")
    loader.rm_dataset()
    loader.set_path("data/test_tmp")
    loader._create_path()
    loader.write()

    loader.move_dataset("data/test_dataset")
    loader_other.set_path("data/test_dataset")
    loader.copy_dataset("data/test_copy")
    DataFormat.merge_dataset([loader, loader_other], "data/test_merge")


def test_add_data():
    """Test add instructions."""
    loader = simple_loader()
    solution_df = loader.df.copy()

    ##########
    # add_ID #
    ##########
    ID = "added_ID"
    feature = "added_feature"

    d = {
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
        "added_feature": np.hstack((list(range(15, 19)), np.full(6, np.nan))),
    }
    df = pd.DataFrame(data=d)  # added_ID DataFrame

    # ignore and overwrite
    d_ID = {
        "timestamp": list(map(str, range(0, 5))),
        "feature0": list(range(5)),
        "feature1": list(range(10, 15)),
    }
    df_ID = pd.DataFrame(data=d_ID)  # DataFrame with different data for ID

    loader.add_ID(df_ID, ID=ID, collision="ignore")
    assert same_data(loader.df, solution_df)
    loader.add_ID(df_ID.copy(), ID=ID, collision="overwrite")
    assert not same_data(loader.df, solution_df)
    loader.add_ID(df.copy(), ID=ID, collision="overwrite")
    assert same_data(loader.df, solution_df)

    # append
    d1 = {
        "timestamp": list(map(str, range(0, 4))),
        "feature0": list(range(4)),
        "feature1": list(range(10, 14)),
        "added_feature": list(range(15, 19)),
    }
    d2 = {
        "timestamp": list(map(str, range(4, 10))),
        "feature0": list(range(4, 10)),
        "feature1": list(range(14, 20)),
        "added_feature": np.full(6, np.nan),
    }
    df1 = pd.DataFrame(data=d1)  # first half of the data
    df2 = pd.DataFrame(data=d2)  # second half of the data

    loader.add_ID(df1.copy(), ID=ID, collision="overwrite")
    loader.add_ID(df2.copy(), ID=ID, collision="append")
    assert same_data(loader.df, solution_df)
    loader.add_ID(df1.copy(), ID=ID, collision="overwrite")
    loader.add_ID(df2.copy(), ID=ID, collision="update")
    assert same_data(loader.df, solution_df)

    ###############
    # add_feature #
    ###############
    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}
    d_feature_other = {
        "timestamp": list(map(str, range(4))),
        feature: list(range(10, 14)),
    }
    df_feature = pd.DataFrame(data=d_feature)  # feature DataFrame
    df_feature_other = pd.DataFrame(
        data=d_feature_other
    )  # DataFrame with different data for feature

    loader.add_feature(df_feature_other.copy(), ID=ID, feature=feature)
    assert not same_data(loader.df, solution_df)
    loader.add_feature(df_feature.copy(), ID=ID, feature=feature)
    assert same_data(loader.df, solution_df)


def test_rm_data():
    """Test add instructions."""
    loader = simple_loader()
    ID = "added_ID"
    feature = "added_feature"

    # Initially present
    assert ID in loader.df.index
    assert feature in loader.df.columns

    # removed
    loader.rm_ID(ID)
    assert ID not in loader.df.index
    loader.rm_feature(feature)
    assert feature not in loader.df.columns


def test_metadata_operations():
    """Test with metadata."""
    loader = simple_loader()

    # add
    # list or no list input
    loader.add_metadata(test_metadata=1)
    assert loader.metadata["test_metadata"][0] == [1]
    loader.add_metadata(test_metadata=[1])
    assert loader.metadata["test_metadata"][0] == [1]
    # different value add
    loader.add_metadata(test_metadata=2)
    loader.add_metadata(test_metadata=[3])
    assert loader.metadata["test_metadata"][0] == [1, 2, 3]

    # overwrite
    # list or no list input
    loader.overwrite_metadata(test_metadata=1)
    assert loader.metadata["test_metadata"][0] == [1]
    loader.overwrite_metadata(test_metadata=[1])
    assert loader.metadata["test_metadata"][0] == [1]

    # set datatype to call all the metadata initialization
    loader.set_datatype("test")


def test_complex_interactions():
    """Test interactions."""
    loader = simple_loader()
    solution_df = loader.df.copy()
    ID = "added_ID"
    feature = "added_feature"

    d_ID = {
        "ID": np.hstack((["name1" for _ in range(5)], ["name2" for _ in range(5)])),
        "timestamp": list(map(str, range(0, 10))),
        "feature0": list(range(10)),
        "feature1": list(range(10, 20)),
        "added_feature": np.hstack((list(range(15, 19)), np.full(6, np.nan))),
    }

    d1 = {
        "timestamp": list(map(str, range(0, 5))),
        "feature0": list(range(5)),
        "feature1": list(range(10, 15)),
    }

    d2 = {
        "timestamp": list(map(str, range(5, 10))),
        "feature0": list(range(5, 10)),
        "feature1": list(range(15, 20)),
    }

    d_feature = {"timestamp": list(map(str, range(4))), feature: list(range(15, 19))}

    df_ID = pd.DataFrame(data=d_ID)  # added_ID DataFrame
    df1 = pd.DataFrame(data=d1)  # name1 DataFrame
    df2 = pd.DataFrame(data=d2)  # name2 DataFrame
    df_feature = pd.DataFrame(data=d_feature)  # added_feature DataFrame

    loader.add_ID(df_ID.copy(), ID=ID, collision="overwrite")
    loader.add_ID(df1.copy(), ID="name1", collision="overwrite")
    loader.add_ID(df2.copy(), ID="name2", collision="overwrite")
    assert same_data(loader.df, solution_df)

    loader.add_ID(df1.copy(), ID="name2", collision="update")
    loader.add_ID(df2.copy(), ID="name1", collision="update")

    loader.add_feature(df_feature.copy(), ID="name1", feature=feature)
    loader.add_feature(df_feature.copy(), ID="name2", feature=feature)

    same_data(loader.df.loc["name1"], solution_df.loc["added_ID"])
    same_data(loader.df.loc["name2"], solution_df.loc["added_ID"])
    same_data(loader.df.loc["added_ID"], solution_df.loc["added_ID"])
