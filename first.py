import pandas as pd
import os

#merge specific files
csv_files = [
    "train/pre_owned_house_transactions.csv",
    "train/pre_owned_house_transactions_nearby_sectors.csv"
]

dfs = [pd.read_csv(f) for f in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)

print(merged_df.head())