
# https://towardsdatascience.com/pandas-groupby-a-simple-but-detailed-tutorial-314b8f37005d
# https://towardsdatascience.com/accessing-data-in-a-multiindex-dataframe-in-pandas-569e8767201d
# https://towardsdatascience.com/summarizing-data-with-pandas-crosstab-efc8b9abecf
# https://towardsdatascience.com/how-to-flatten-multiindex-columns-and-rows-in-pandas-f5406c50e569
# https://datascientyst.com/list-aggregation-functions-aggfunc-groupby-pandas/
# https://stackoverflow.com/questions/25929319/how-to-iterate-over-pandas-multiindex-dataframe-using-index
# https://stackoverflow.com/questions/24495695/pandas-get-unique-multiindex-level-values-by-label
# https://stackoverflow.com/questions/55706391/pandas-crosstab-on-multiple-columns-then-groupby

# https://matplotlib.org/stable/gallery/pyplots/pyplot_text.html#sphx-glr-gallery-pyplots-pyplot-text-py


import pandas as pd
import numpy as np
from myUtils import pickle_load, pickle_dump

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"

# # filename_symbols = path_data_dump + 'vg_symbols_4chars_max.csv'  # symbols text file
# filename_symbols = path_data_dump + 'my_symbols.csv'  # symbols text file

# _filename_pickled_df_OHLCVA_downloaded = 'df_OHLCVA_downloaded '  # OHLCVA downloaded from Yahoo
filename_pickled_df_adjOHLCV = 'df_adjOHLCV'  # adjusted OHLCV
filename_pickled_df_symbols_close = "df_symbols_close"  # symbols' adjusted close
filename_pickled_symbols_df_adjOHLCV =  'symbols_df_adjOHLCV'  # symbols in df_adjOHLCV
filename_pickled_perf_rank_dict =  'perf_rank_dict'  # store symbols from performance rank results
filename_pickled_r_all_ranks =  'r_all_ranks'  # list of top 100 most common symbols from performance rank results
filename_pickled_df_a = 'df_OHLCV_clean'  # df adjusted OHLCV, dropped symbols with no vol and close
filename_pickled_df_c = 'df_close_clean'  # df close, dropped symbols with no vol and close

verbose = False  # True prints more output

#################
# look_back_days = -250 * 60  # subset df iloc days
look_back_days = -250 * 6  # subset df iloc days, 6 years of data
#################


print(f"Full path to pickled df_symbols_close:  {path_data_dump}{filename_pickled_df_symbols_close}")
df_close = pickle_load(path_data_dump, filename_pickled_df_symbols_close, verbose=verbose)
print(f"Full path to pickled df_adjOHLCV:  {path_data_dump}{filename_pickled_df_adjOHLCV}")
df_adjOHLCV = pickle_load(path_data_dump, filename_pickled_df_adjOHLCV, verbose=verbose)


# https://stackoverflow.com/questions/63826291/pandas-series-find-column-by-value
df = df_adjOHLCV[look_back_days::]
df_v = df.xs('Volume', level=1, axis=1)  # select only Volume columns
rows, cols = np.where(df_v == 0)  # row index, column index where trading volumes are zero
idx_no_volume = list(set(cols))
idx_no_volume.sort()
symbols_no_volume = df_v.columns[idx_no_volume]
print(f'symbols with no volume:\n{symbols_no_volume}')


df_dif = df_v - df_v.shift(periods=1)
rows, cols = np.where(df_dif == 0)
idx_same_volume = list(set(cols))
idx_same_volume.sort()
idx_same_volume
symbols_same_volume = df_v.columns[idx_same_volume]
print(f'symbols with same volume:\n{symbols_same_volume}')


df_c = df.xs('Close', level=1, axis=1)  # select only Close columns
df_c = df_c.fillna(0).copy()  # convert NaNs to zeros
rows, cols = np.where(df_c == 0)  # row index, column index where trading volumes are zero
idx_no_close = list(set(cols))
idx_no_close.sort()
symbols_no_close = df_c.columns[idx_no_close]
print(f'symbols with NaN close:\n{symbols_no_close}')


symbols_drop = list(symbols_no_close) + list(symbols_no_volume) + list(symbols_same_volume) # combine symbols with no volume and no close
print(f'combined symbols with no volume, same volume and no close, inculdes duplicate symbols: {len(symbols_drop)}')
symbols_drop = list(set(symbols_drop))  # drop duplicate symbols
symbols_drop .sort()
df_a = df.drop(symbols_drop, axis=1, level=0)  # drop symbols from OHLCA df
df_c = df_close.iloc[look_back_days::]
df_c = df_c.drop(symbols_drop, axis=1)
print(f'unique symbols dropped from df_a (adjOLHLV) and df_c (Close): {len(symbols_drop)}')


print(f'symbols with no volume:   {len(symbols_no_volume):>5,}')
print(f'symbols with same volume: {len(symbols_same_volume):>5,}')
print(f'symbols with no close:    {len(symbols_no_close):>5,}\n')
print(f'symbols total before drop:                                        {len(df_close.columns):>5,}')
print(f'unique symbols dropped from df OHLCVA (df_a) and df Close (df_c): {len(symbols_drop):>5,}\n')
print('                                          symbols     rows')
print(f'df adjOHLCV (df_a) after dropped symbols:   {len(df_a.columns)/5:>5,.0f}    {len(df_a):>5,}')
print(f'df Close (df_c) after dropped symbols:      {len(df_c.columns):>5,}    {len(df_c):>5,}')



pickle_dump(df_a, path_data_dump, filename_pickled_df_a)
print(f'pickled df adjOHLCV after dropping symbols with no volume, same volume, and no close:\n{path_data_dump}{filename_pickled_df_a}')
pickle_dump(df_c, path_data_dump, filename_pickled_df_c)
print(f'pickled df Close after dropping symbols with no volume, same volume, and no close:\n{path_data_dump}{filename_pickled_df_c}')


from myUtils import list_dump

f_symbols_df_close_clean = 'symbols_df_close_clean.csv'  # symbols text file
symbols_df_c = list(df_c)  # column names in df_c
list_dump(symbols_df_c, path_data_dump, f_symbols_df_close_clean)# df_c.columns.to_csv(f_symbols_df_close_clean)


df_a.tail()


df_c.tail()


