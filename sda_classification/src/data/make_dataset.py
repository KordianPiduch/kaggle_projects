# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import src.helping_functions as hf


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    # load and prepare dataset part1
    df1 = pd.read_csv(input_filepath + "/df1.csv", index_col=0)
    logger.info('df1.csv loaded')

    to_drop = ["EmployeeCount"]
    to_int = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked"] 
    to_category = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", 
    "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction"]

    hf.clean_dataframe(df1, to_drop, to_int, to_category)
    logger.info('df1.csv cleaned')

    # load and prepare dataset part1
    df2 = pd.read_csv(input_filepath + "/df2.csv", index_col=0)
    logger.info('df2.csv loaded')

    to_drop = ["Over18"]
    to_int = ["PercentSalaryHike", "StandardHours", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", 
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "YearlyIncome"] 
    to_category = ["OverTime", "Attrition", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"]

    hf.clean_dataframe(df2, to_drop, to_int, to_category)
    logger.info('df2.csv cleaned')

    # this will be the target for the model
    df2["Attrition"] = df2["Attrition"].replace(["No", "Yes"], [0, 1])

    # save cleaned dataframe
    full_df = pd.concat([df1, df2], axis=1)
    full_df.to_csv(output_filepath + "/full_df.csv")
    logger.info('full_df.csv saved in /data/processed')
    full_df.to_pickle(output_filepath + "/full_df")
    logger.info('pickle full_df saved in /data/processed')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
