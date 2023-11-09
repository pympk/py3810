import pandas as pd
import pandas_market_calendars as mcal
from myUtils import pickle_load, pickle_dump

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 900)

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"

# pickle file of past picks
fp_df_picks = 'df_picks'
# pickle file of NYSE dates missing from df_picks
fp_NYSE_dates_missing_in_df_picks = 'NYSE_dates_missing_in_df_picks'


def compare_lists(list_a, list_b):
  """Compares two lists and returns a list of values
     that are in list A but not in list B.

  Args:
    list_a: A list of objects.
    list_b: A list of objects.

  Returns:
    A list of values that are in list_a but not in list_b.
  """

  list_difference = []

  for item in list_a:
    if item not in list_b:
      list_difference.append(item)

  return list_difference


df_picks = pickle_load(path_data_dump, fp_df_picks)
# drop duplicates
df_picks = \
  df_picks.drop_duplicates(subset=['date_end_df_train',
                                   'max_days_lookbacks',
                                   'days_lookbacks'],
                           keep='last')
# sort, most recent date is first
df_picks = \
  df_picks.sort_values(by=['date_end_df_train',
                           'max_days_lookbacks', 'days_lookbacks'],
                       ascending=False)
# re-index
df_picks = df_picks.reset_index(drop=True)
# save results
pickle_dump(df_picks, path_data_dump, fp_df_picks)
print(f'df_picks, len({len(df_picks)}):\n{df_picks}')

start_date = df_picks.date_end_df_train.min()
end_date = df_picks.date_end_df_train.max()
print(f'df_picks start date: {start_date}')
print(f'df_picks end date: {end_date}')

l_dates_df_picks = \
  df_picks.date_end_df_train.unique().tolist()  # unique dates in df_picks
print(f'l_dates_df_picks, len({len(l_dates_df_picks)}):\n{l_dates_df_picks}')

nyse = mcal.get_calendar('NYSE')
# NYSE dates from df_picks start date to end date
dates_NYSE = \
  nyse.valid_days(start_date=start_date,
                  end_date=end_date).strftime('%Y-%m-%d')
# print(f'len(dates_NYSE): {len(dates_NYSE)}')
dates_NYSE_reversed_sorted = dates_NYSE.sort_values(ascending=False)
print('NYSE dates from df_picks start date to end date')
print(f'dates_NYSE_reversed_sorted, len({len(dates_NYSE)}):\n{dates_NYSE_reversed_sorted}')

NYSE_dates_missing_in_df_picks = compare_lists(dates_NYSE_reversed_sorted, l_dates_df_picks)
NYSE_dates_missing_in_df_picks.sort(reverse=True)  # sorted inplace, newest first
pickle_dump(NYSE_dates_missing_in_df_picks, path_data_dump, fp_NYSE_dates_missing_in_df_picks)
print(f'NYSE_dates_missing_in_df_picks is saved in {path_data_dump}{fp_NYSE_dates_missing_in_df_picks}')
print(f'NYSE_dates_missing_in_df_picks, (len={len(NYSE_dates_missing_in_df_picks )}):\n{NYSE_dates_missing_in_df_picks }')
