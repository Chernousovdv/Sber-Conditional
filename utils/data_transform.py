import pandas as pd


def take_diff(df, k=1):
    """
    Take k-order difference for each column in the DataFrame.
    """
    df_diffed = df.copy()
    for col in df_diffed.columns:
        df_diffed[col] = df_diffed[col].diff(periods=k)

    return df_diffed[k:]
