"""Minimal OHE helper matching kedro_carprice_prediction.data_prep.encodeCatVariables."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_cat_variables(
    df: pd.DataFrame,
    ohe_cols: list[str],
    ohe_encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    df_encoded = df.copy()
    cols_present = [c for c in ohe_cols if c in df_encoded.columns]
    if not cols_present:
        return df_encoded, ohe_encoder

    if ohe_encoder is None:
        ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = ohe_encoder.fit_transform(df_encoded[cols_present])
    else:
        encoded = ohe_encoder.transform(df_encoded[cols_present])

    feature_names = ohe_encoder.get_feature_names_out(cols_present)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_encoded.index)
    df_encoded = df_encoded.drop(columns=cols_present)
    return pd.concat([df_encoded, encoded_df], axis=1), ohe_encoder
