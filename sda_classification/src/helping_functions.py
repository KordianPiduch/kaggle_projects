import pandas as pd

 
def change_type(dataframe: pd.DataFrame, col_list: list(), new_type: str):
    for type in col_list:
        dataframe[type] = dataframe[type].astype(new_type)