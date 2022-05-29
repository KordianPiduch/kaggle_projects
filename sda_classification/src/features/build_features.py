import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


df = pd.read_pickle("../data/processed/full_df")

# select target and features 
target = "Attrition"  # nan values 
features = [
    'Age', 
    'DailyRate', 
    # 'DistanceFromHome', 
    'JobLevel', 
    'JobSatisfaction', 
    'MonthlyIncome',
    # 'NumCompaniesWorked', 
    'OverTime', 
    'PercentSalaryHike', 
    # 'TotalWorkingYears'
    ]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# fill na with most frequent value
y_train = y_train.fillna(y_train.value_counts().index[0])
y_test = y_test.fillna(y_test.value_counts().index[0])

# build pipelines
num_attr = X.select_dtypes(include='number').columns
cat_attr = X.select_dtypes(exclude='number').columns

numeric_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='median')),
        ("std_scaler", StandardScaler()),
    ]
)

category_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('ohe', OneHotEncoder())
    ]
)

full_pipeline = ColumnTransformer(
    [
        ("numerical", numeric_pipeline, num_attr),
        ("categorical", category_pipeline, cat_attr)
    ]
)

X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)