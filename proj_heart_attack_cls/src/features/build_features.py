import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer



def get_train_test(path: str, test_size=0.2):

    df = pd.read_pickle(path)

    features = df.drop('output', axis=1).columns
    target = 'output'

    X = df[features]
    y = df[target]

    num_attr = X.select_dtypes(include='number').columns
    cat_attr = X.select_dtypes(exclude='number').columns

    # split dataset for train and test test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)

    # pipeline for numeric attributes
    num_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    # pipeline for categorial attributes
    cat_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(sparse=False))
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ('num', num_pipeline, num_attr),
            ('cat', cat_pipeline, cat_attr)
        ]
    )

    # train df with column names after pipeline preprocessing 
    X_train_tr = pd.DataFrame(data=full_pipeline.fit_transform(X_train), columns=full_pipeline.get_feature_names_out())

    # test df with column names after pipeline preprocessing 
    X_test_tr = pd.DataFrame(data=full_pipeline.transform(X_test), columns=full_pipeline.get_feature_names_out())

    return (X_train_tr, X_test_tr, y_train, y_test)