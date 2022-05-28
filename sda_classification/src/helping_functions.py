import pandas as pd

 
def change_type(dataframe: pd.DataFrame, col_list: list(), new_type: str):
    for type in col_list:
        dataframe[type] = dataframe[type].astype(new_type)


def clean_dataframe(dataframe: pd.DataFrame, drop: list, to_int: list, to_category: list):
    dataframe.drop(columns=drop, inplace=True)
    change_type(dataframe, to_int, "int")
    change_type(dataframe, to_category, "category")