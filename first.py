import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

#based on the sample submission, we need some combination of month and sector for new house targets
#some explanatory data analysis

def clean_read_csv(path):
    """
    This function loads in csv files and strips leading and trailing whitespaces
    Returns the cleaned csv file
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_month(date_col):
    """
    This function gets the month from the date column
    """
    s = pd.to_datetime(date_col, errors="coerce", format="%Y %b")
    #may need to add error checking here
    return s.dt.to_period("M").dt.to_timestamp()

def loading_training_csv_tables():
    """
    This function loads the training csv files into a pandas dataframe, modifying the column names
    """
    #POI
    poi = clean_read_csv("train/sector_POI.csv")

    #new house transactions
    new_house_trans = clean_read_csv("train/new_house_transactions.csv")
    new_house_trans["month"] = get_month(new_house_trans["month"])

    #new house transactions nearby sectors
    new_house_trans_near = clean_read_csv("train/new_house_transactions_nearby_sectors.csv")
    new_house_trans_near["month"] = get_month(new_house_trans_near["month"])

    #preowned house transactions
    preowned_house_trans = clean_read_csv("train/pre_owned_house_transactions.csv")
    preowned_house_trans["month"] = get_month(preowned_house_trans["month"])

    #preowned houses nearby
    preowned_house_trans_near = clean_read_csv("train/pre_owned_house_transactions_nearby_sectors.csv")
    preowned_house_trans_near["month"] = get_month(preowned_house_trans_near["month"])

    #csv for land transactions
    land = clean_read_csv("train/land_transactions.csv")
    land["month"] = get_month(land["month"])
