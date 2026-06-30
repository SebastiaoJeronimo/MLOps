"""
This is a boilerplate pipeline 'split_train'
generated using Kedro 1.3.1
"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_val(df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42):
    """Hold out a validation split from the LABELED training data, before
    any cleaning/leakage-sensitive steps. car_test_raw (the unlabeled batch)
    is untouched -- this only splits the labeled set for offline evaluation."""
    
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

