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

#function to load training data and clean month column
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

    #nearby land transactions
    land_near = clean_read_csv("train/land_transactions_nearby_sectors.csv")
    land_near["month"] = get_month(land_near["month"])

    #city data
    city_idx = clean_read_csv("train/city_indexes.csv")

    #city search index
    city_search_idx = clean_read_csv("train/city_search_index.csv")
    city_search_idx["month"] = get_month(city_search_idx["month"])
    #group data by month, add up search volume across all keywords and sources
    #rename the columns to illustrate total monthly search volume
    search_monthly = (
        city_search_idx.groupby("month", as_index=False)["search_volume"]
        .sum()
        .rename(columns={"search_volume": "search_volume_total"})
    )

    result = {
        "new_house": new_house_trans,
        "new_house_nb": new_house_trans_near,
        "pre_owned": preowned_house_trans,
        "pre_owned_nb": preowned_house_trans_near,
        "land": land,
        "land_nb": land_near,
        "poi": poi,
        "city_idx": city_idx,
        "search_monthly": search_monthly
    }
    return result

#function to build balanced data, filling missing values with 0
def build_clean_data(new_houses):
    new_house = new_houses.copy()
    sectors = new_house["sector"].unique()
    monmin, monmax = new_house["month"].min(), new_house["month"].max()
    months = pd.date_range(start=monmin, end=monmax, freq="M").to_timestamp()

    panel = pd.MultiIndex.from_product([sectors, months], names=["sector", "month"]).to_frame(index=False)

    keep_cols = [
        "month", "sector",
        "num_new_house_transactions",
        "area_new_house_transactions",
        "price_new_house_transactions",
        "amount_new_house_transactions",
        "area_per_unit_new_house_transactions",
        "total_price_per_unit_new_house_transactions",
        "num_new_house_available_for_sale",
        "area_new_house_available_for_sale",
        "period_new_house_sell_through"
    ]

    new_house_small = new_house[[c for c in keep_cols if c in new_house.columns]].copy()
    df = panel.merge(new_house_small, on="month", how="left")
    df["amount_new_house_transactions"] = df["amount_new_house_transactions"].fillna(0)

    return df

def merge_features(panel, tables):
    df = panel.copy()

#main function
def main():
    """
    Main function doing all the work
    """