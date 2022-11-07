---
author: Francis Huot-Chantal
---

# Installation

``` shell
python -m pip install -e .
```

# Usage

Use exactly one loader per datatype.

## Simple example

notebook/example_add.org

## add_datatype

## add_ID

## add_feature

## Performance choices

1.  It is recommended to add timeseries with the method `add_ID`.

2.  `add_feature` is not a performing method

3.  `autowrite=False`, by default. It means, it does

not write to file automatically after operations. Don't forget to
`write()` when finish to save to file.

## More examples

See the notebooks

## How the data is stored

The collection of all the data is called the dataset.

The `datatype` refers type of the data which informs about the structure
of the data. A given `datatype` as the exact same `datafeature` which is
the name their features (observations). `Datatype` is a collection of
multiple categories of input from different `ID`.

In `pandas` term, `ID` is the column with the same name and `feature` is
the name of the rest of the columns. The `datatype` is store in a
sequence of `split`.

First example,

``` example
datatype = "simulated"
split = [""]
ID = ["ARMA1", "ARMA2", "RandomForest"]
features = ["returns"]
```

Second example,

``` example
datatype = "Stocks"
split = ["date1", "date2"]
ID = ["ABC", "XYZ"]
features = ["side", "quantity", "price"]
```

Note that the data is separated on two files (days) because of the
quantity of data. Otherwise only one file would be needed

## The splits

Note that the data is separated on two files (days) because of the
quantity of data. Otherwise only one file would be needed

Split_names is an array of names Split_index is the current index to
load in split_names

## The DataFrames

### Data

## The DataFrames

In `pandas` term, `ID` is the column with the same name and
`datafeature` is the name of the rest of the columns. The `datatype` is
store in a sequence of `filenames`.

|     |          |          |
|-----|----------|----------|
| ID  | feature1 | feature2 |
|     |          |          |

`datafeature` = \[\]

Data is index using (ID, timestamp)

|                |                |                |     |                |
|----------------|----------------|----------------|-----|----------------|
| feature n: any |                |                |     |                |
| ID: str        | timestamp: str | feature 1: any | …   | feature n: any |

### Metadata

It is important to have full control over metadata. Metadata is not
affected by the attribute overwrite.

Metadata index using datatype

List elements in metadata are unique. The use of a set was consider but
was restrictive whenever a user would want to loop over some metadata in
a predetermined way.

| datatype: str | split: list\[str\] | IDs: list\[str\] | features: list\[str\] | arbitrary_metadata: any |
|---------------|--------------------|------------------|-----------------------|-------------------------|

Arbitrary metadata example:

| start | frequency | nb_observation |
|-------|-----------|----------------|

# Parallel execution

Parallel attribute must be set to True if the loader is intended to be
used in parallel. The metadata output will therefore be base on loader
datatype and split. To generate the `metadata.pqt`, the method
`merge_metadata` must be used after all the parallel processes are
completed.
