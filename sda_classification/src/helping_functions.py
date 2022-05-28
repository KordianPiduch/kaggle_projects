import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# custom transformer
class TrimOutliers(BaseEstimator, TransformerMixin):
    columns_to_transform = ["Age", "DistanceFromHome", "TotalWorkingYears", "YearsAtCompany", 
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

    def __init__(self, limit:int=100):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in self.columns_to_transform:
            X.loc[(X[column] > self.limit), column] = X[column].median()

 
def change_type(dataframe: pd.DataFrame, col_list: list(), new_type: str):
    for type in col_list:
        dataframe[type] = dataframe[type].astype(new_type)


def clean_dataframe(dataframe: pd.DataFrame, drop: list, to_int: list, to_category: list):
    dataframe.drop(columns=drop, inplace=True)
    change_type(dataframe, to_int, "int")
    change_type(dataframe, to_category, "category")


