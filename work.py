import pandas as pd
from myUtils import pickle_load

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 11)
pd.set_option("display.max_colwidth", 16)
pd.set_option("display.width", 145)

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"

fp_df_close_clean = "df_close_clean"
fp_dates_missing_in_df_picks = "dates_missing_in_df_picks"


def drop_dates_after_input_date(df, input_date):
    """Drops dates after the input date in a Pandas DataFrame with
       a Date index.

    Args:
      df: A Pandas DataFrame with a Date index.
      input_date: A datetime object.

    Returns:
      A Pandas DataFrame with the dates after the input date dropped.
    """

    # Get the index of the input date in the DataFrame.
    input_date_index = df.index.get_loc(input_date)

    # Drop the rows after the input date.
    df = df.iloc[: input_date_index + 1]

    return df

# TODO change to drop_last_n_rows 

def drop_dates_after_input_date(df, input_date):
    """Drops dates after the input date in a Pandas DataFrame with
       a Date index.

    Args:
      df: A Pandas DataFrame with a Date index.
      input_date: A datetime object.

    Returns:
      A Pandas DataFrame with the dates after the input date dropped.
    """

    # Get the index of the input date in the DataFrame.
    input_date_index = df.index.get_loc(input_date)

    # Drop the rows after the input date.
    df = df.iloc[: input_date_index + 1]

    return df


df_close_clean = pickle_load(path_data_dump, fp_df_close_clean)
print(f"df_close_clean:\n{df_close_clean}")

dates_missing_in_df_picks = \
  pickle_load(path_data_dump, fp_dates_missing_in_df_picks)
print(
    f"dates_missing_in_df_picks, len({len(dates_missing_in_df_picks)}):\n{dates_missing_in_df_picks}"
)

for i, date in enumerate(dates_missing_in_df_picks):
    df_ = drop_dates_after_input_date(df=df_close_clean, input_date=date)
    print(f"i: {i}, len({len(df_)})\n{df_.tail(2)}")
