# %%
# df_adjOHLCV has data only for NYSE trading days, no weekend data
# df_adjOHLCV includes data for weekends when BTC trades
# read symbols in file to list syms_in_file
# download OHLCV data for symbols in syms_in_file
# drop symbols with all NaN in OHLCV columns from df
# rename column names from ['Open', ..., 'Volume'] to ['open', ..., 'volume']
# drop weekend data by reindex to date index of index_symbol
# pickled df_adjOHLCV
# pickled df_adjOHLCV
# pickled symbols_df_adjOHLCV
# create df_symbols_close, sort df by symbols
# pickled df_symbols_close

# %%
# https://stackoverflow.com/questions/67690778/how-to-detect-failed-downloads-using-yfinance
import yfinance as yf
import yfinance.shared as shared
import time
import pandas as pd
# from datetime import date, timedelta, datetime
from myUtils import pickle_dump, pickle_load, read_symbols_file # NOQA
from myUtils import drop_symbols_all_NaN, chunked_list # NOQA
from myUtils import yf_download_AdjOHLCV_noAutoAdj
verbose = False  # True prints more output

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"

# filename_symbols = path_data_dump + 'vg_symbols_4chars_max.csv'  # symbols text file
# filename_symbols = path_data_dump + 'vg_symbols_4chars_max_clean.csv'  # symbols text file
filename_symbols = path_data_dump + 'my_symbols.csv'  # symbols text file

filename_pickled_df_OHLCVA_downloaded = 'df_OHLCVA_downloaded '  # OHLCVA downloaded from Yahoo
filename_pickled_df_adjOHLCV = 'df_adjOHLCV'  # adjusted OHLCV
filename_pickled_df_symbols_close = "df_symbols_close"  # symbols' adjusted close
filename_pickled_symbols_df_adjOHLCV =  'symbols_df_adjOHLCV'  # symbols in df_adjOHLCV

# %%
# function to adjust OHLCV
# https://github.com/ranaroussi/yfinance/issues/687
def auto_adjust(data):
    df = data.copy()
    ratio = df["Close"] / df["Adj Close"]
    df["Adj Open"] = df["Open"] / ratio
    df["Adj High"] = df["High"] / ratio
    df["Adj Low"] = df["Low"] / ratio

    df.drop(
        ["Open", "High", "Low", "Close"],
        axis=1, inplace=True)

    df.rename(columns={
        "Adj Open": "Open", "Adj High": "High",
        "Adj Low": "Low", "Adj Close": "Close"
    }, inplace=True)

    # df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df[["Open", "High", "Low", "Close", "Volume"]]

def close_vs_Yahoo (symbols, df):
  for symbol in symbols:
    s = df.iloc[-250]
    
    sDate = s.name.strftime('%Y-%m-%d')
    sClose = s[symbol].Close

    df_sym = yf.Ticker(symbol).history(period='2y')
    # 10:45PM got Adj Close
    yhClose = df_sym.loc[sDate]['Close']

    abs_pct_diff = abs(1 - sClose/yhClose)*100
    print(f'symbol:  {symbol:>4}   Date: {sDate:13}df_Close: {sClose:>10.5f} \
    Yahoo_Close: {yhClose:>10.5f}   %_dif_Close: {abs_pct_diff:>7.5f}')
    if abs_pct_diff > .0001:
      msg_err = f'{symbol}  %_dif_Close > .0001'
      # raise SystemExit(msg_err)
      print(msg_err)  # even if discrepancy, the relative relationship of OHLCV in df_close matches Yahoo_Close
    if symbol == symbols[-1]:
      msg_done = f"No errors found.  Symbols' 'Adjusted Close' matched Yahoo symbols' Close "
      print(msg_done)


# %%
# Stop if last date in df_adjOHLCV is the same as Yahoo download
index_symbol = "XOM"  
df_XOM = yf.download(index_symbol)
download_last_date = df_XOM.index[-1].strftime('%Y-%m-%d')
print(f"Full path to pickled df_adjOHLCV:  {path_data_dump}{filename_pickled_df_adjOHLCV}")
print(f'download_last_date: {download_last_date}')
df = pickle_load(path_data_dump, filename_pickled_df_adjOHLCV, verbose=verbose)
df_adjOHLCV_last_date = df.index[-1].strftime('%Y-%m-%d')
print(f'df_adjOHLCV_last_date: {df_adjOHLCV_last_date}')
if download_last_date == df_adjOHLCV_last_date:
  msg_stop = f"df_adjOHLCV is up-to-date,\nlast date from df_adjOHLCV is {df_adjOHLCV_last_date},\nlast date from Yahoo's {index_symbol} download is {download_last_date}"
  raise SystemExit(msg_stop )

# %%
# verify df test_symbols' close against Yahoo
# even if discrepancy, the relative relationship of OHLCV in df_close matches Yahoo_Close
test_symbols = ['A', 'LMT', 'SPY']
_df_list=[]
_df = yf_download_AdjOHLCV_noAutoAdj(test_symbols, verbose=False)
_df_list.append(_df)
_df = pd.concat(_df_list, axis=1)
# get unique symbols from column name
_l_symbols = list(set(_df.columns.get_level_values(0)))
_l_symbols.sort()

_df_adj_list=[]
# adjust OHLC using 'Adj Close'
for symbol in _l_symbols:
  _df1 = auto_adjust(_df[symbol])
  _df_adj_list.append(_df1)

_df_adj = pd.concat(_df_adj_list, axis=1)
_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# create multilevel column names
_col_names = pd.MultiIndex.from_product([_l_symbols, _cols])
_df_adj.columns = _col_names

close_vs_Yahoo(test_symbols, _df_adj)

# %%
# read symbols in file to a list and convert to chunks
#  each chunk must have more than 1 symbol, otherwise, symbol won't appear as column name
symbols_in_file = read_symbols_file(filename_symbols)
print(f'symbols in file: {len(symbols_in_file)}')
# TODO add list of SPY, VGT, FTSM, QQQ, BNDX 
symbols_add = ['SPY', 'VGT', 'FTEC', 'QQQ', 'VWO', 'VTV', 'BNDX', 'USO', 'JJC', 'BCI', 'NG', 'GBTC', 'BTC-USD', 'ETH-USD']
print(f'symbols added:   {len(symbols_add)}')
symbols = symbols_in_file + symbols_add
# https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python
symbols = list(set(symbols))  # keep unique symbols
symbols.sort()  # sort unique symbols in alphabetical order
print(f'unique symbols:  {len(symbols)}')
symbols_chunks = chunked_list(symbols, 400)  # e.g. [['A', 'BB', ...], ['CC', 'DD', ...], ..., ['Z', 'ZWS', ...]]
for chunk in symbols_chunks:
  if len(chunk) == 1:
    raise SystemExit(f'ERROR, Must be more than 1 symbol in each chunk\nsymbols in chunk: {chunk}')

# %%
df_list=[]
symbols_download_err = []
# took 24 minutes to download 1917 symbols without error caused by Yahoo
# for i, symbols in enumerate(symbols_chunks):
for i, symbols in enumerate(symbols_chunks):
  df = yf_download_AdjOHLCV_noAutoAdj(symbols, verbose=False)  
  symbols_download_err.append(list(shared._ERRORS.keys()))
  # print(f'symbols_download_err: {symbols_download_err}')
  print(f'\nsymbols_download_err: {symbols_download_err}')  
  df_list.append(df)
  # pause between download
  if i < len(symbols_chunks) - 1 :  # skip pause after last download
    print(f'downloaded symbols from chuck {i}, sleep start')
    # sleep 78(18m 25s)
    time.sleep(78)
    print(f'downloaded symbols from chuck {i}, sleep ends')
  else:
    print(f'downloaded symbols from all chucks')

print(f'symbols_download_err: {symbols_download_err}')

# %%
# flatten symbols_download_err which is a list-of-lists
l_symbols_err = [val for sublist in symbols_download_err for val in sublist]
pickle_dump(l_symbols_err, path_data_dump, 'l_symbols_err')
l_symbols_err

# %%
# if l_symbols_err:  # re-download symbols with download error 
#   symbols_download_err = []
#   # took 24 minutes to download 1917 symbols without error caused by Yahoo
#   # for i, symbols in enumerate(symbols_chunks):
#   for i, symbols in enumerate(l_symbols_err):
#     df = yf_download_AdjOHLCV_noAutoAdj(symbols, verbose=False)  
#     symbols_download_err.append(list(shared._ERRORS.keys()))
#     # print(f'symbols_download_err: {symbols_download_err}')
#     print(f'\nsymbols_download_err: {symbols_download_err}')  
#     df_list.append(df)
#     # pause between download
#     if i < len(symbols_chunks) - 1 :  # skip pause after last download
#       print(f'downloaded symbols from chuck {i}, sleep start')
#       # sleep 78(18m 25s)
#       time.sleep(78)
#       print(f'downloaded symbols from chuck {i}, sleep ends')
#     else:
#       print(f'downloaded symbols from all chucks')

#   print(f'symbols_download_err: {symbols_download_err}')

# %%
df = pd.concat(df_list, axis=1)
if l_symbols_err:  # if list is not empty, drop symbols with errors
  print(f'l_symbols_err is empty: {l_symbols_err}')  
  df.drop(l_symbols_err, axis=1, level=0, inplace=True)  # drop symbols with errors

l_symbols = list(set(df.columns.get_level_values(0)))  # get unique symbols
l_symbols.sort()
df = df[l_symbols]  # sort columns by list
print(f"Full path to pickled df_OHLCVA_downloaded:  {path_data_dump}{filename_pickled_df_OHLCVA_downloaded}")
pickle_dump(df, path_data_dump, filename_pickled_df_OHLCVA_downloaded)  # save OHLCA data

# %%
# adjust OHLC
df_adj_list=[]
for symbol in l_symbols:
  _df = auto_adjust(df[symbol])
  df_adj_list.append(_df)

df_adj = pd.concat(df_adj_list, axis=1)
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
col_names = pd.MultiIndex.from_product([l_symbols, cols])  # create multilevel column names
df_adj.columns = col_names  # reindex columns

print(f"Full path to pickled df_adjOHLCV:  {path_data_dump}{filename_pickled_df_adjOHLCV}")
pickle_dump(df_adj, path_data_dump, filename_pickled_df_adjOHLCV)  # save adjusted OHLCV data

# %%
df_adjOHLCV = pickle_load(path_data_dump, filename_pickled_df_adjOHLCV)

# %%
# drop symbols with all NaN in OHLCV columns from df
df_adjOHLCV, symbols_OHLCV, symbols_dropped = drop_symbols_all_NaN(df_adjOHLCV, verbose)
print(f'symbols with all NaN dropped from df_adjOHLCV: {symbols_dropped}')

# %%
# rename columns OHLCV *ONLY AFTER* dropping symbols with all NaN from df,
#   symbols with all NaN has an added AdjClose column and will cause errors  
#  rename column names from ['Open', ..., 'Volume'] to ['open', ..., 'volume']
#  .remove_unused_levels() prevents ValueError
#   e.g ValueError: On level 1, code max (5) >= length of level (5). NOTE: this index is in an inconsistent state
# The error may be caused by removing symbols from the dataframe with all NaN in OHLCV columns
df_adjOHLCV.columns = df_adjOHLCV.columns.remove_unused_levels()
# # set_levels reorders df columns in alphabetical order, so the list of column names also needs to be in alphabetical order
# df_adjOHLCV.columns = df_adjOHLCV.columns.set_levels(['close', 'high', 'low', 'open', 'volume'], level=1)

# %%
# drop weekend data by re-indexing to date-index of index_symbol
myNaN = float('nan')
# use Exxon's date as proxy for NYSE trading dates
df_adjOHLCV = df_adjOHLCV.reindex(df_XOM.index, fill_value=myNaN)

# %%
# pickle df_adjOHLCV and symbols
# print(f"Full path to pickled df_adjOHLCV_downloaded:  {path_data_dump}{filename_pickled_df_adjOHLCV_downloaded}")
# pickle_dump(df_adjOHLCV_downloaded, path_data_dump, filename_pickled_df_adjOHLCV_downloaded, verbose=verbose)
print(f"Full path to pickled df_adjOHLCV:  {path_data_dump}{filename_pickled_df_adjOHLCV}")
pickle_dump(df_adjOHLCV, path_data_dump, filename_pickled_df_adjOHLCV, verbose=verbose)
print(f"Full path to pickled symbols_df_adjOHLCV:  {path_data_dump}{filename_pickled_symbols_df_adjOHLCV}")
pickle_dump(symbols_OHLCV, path_data_dump, filename_pickled_symbols_df_adjOHLCV, verbose=verbose)

# %%
symbols_OHLCV = list(set([i[0] for i in list(df_adjOHLCV)]))
df_symbols_close = df_adjOHLCV.xs('Close', level=1, axis=1)
pickle_dump(df_symbols_close, path_data_dump, filename_pickled_df_symbols_close, verbose=verbose)

# %%
# retrieve pickled files

print(f"Full path to pickled df_OHLCVA_downloaded:  {path_data_dump}{filename_pickled_df_OHLCVA_downloaded}")
df_OHLCVA_downloaded = pickle_load(path_data_dump, filename_pickled_df_OHLCVA_downloaded, verbose=verbose)

print(f"Full path to pickled df_adjOHLCV:  {path_data_dump}{filename_pickled_df_adjOHLCV}")
df_adjOHLCV = pickle_load(path_data_dump, filename_pickled_df_adjOHLCV, verbose=verbose)

print(f"Full path to pickled df_symbols_close:  {path_data_dump}{filename_pickled_df_symbols_close}")
df_close = pickle_load(path_data_dump, filename_pickled_df_symbols_close, verbose=verbose)

print(f"Full path to pickled symbols_df_adjOHLCV:  {path_data_dump}{filename_pickled_symbols_df_adjOHLCV}")
symbols_df = pickle_load(path_data_dump, filename_pickled_symbols_df_adjOHLCV, verbose=verbose)

# %%
print(f'df_OHLCVA_downloaded.info():\n{df_OHLCVA_downloaded.info()}\n')
print(f'df_adjOHLCV.info():\n{df_adjOHLCV.info()}\n')
print(f'df_close.info():\n{df_close.info()}\n')
print(f'len(symbols_df):\n{len(symbols_df)}\n')

# %%
count_symbols = len(symbols_df)
count_df_close_cols = len(df_close.columns)
count_cols_df_OHLCVA_downloaded = len(df_OHLCVA_downloaded.columns)
count_cols_df_adjOHLCV = len(df_adjOHLCV.columns)
if count_symbols != count_df_close_cols:
  raise SystemExit(f'symbol count: {count_symbols} does not equal df_close column count: {count_df_close_cols}')
if count_symbols * 6 != count_cols_df_OHLCVA_downloaded:
  raise SystemExit(f'6 x symbol count: {6 * count_symbols} does not equal df_OHLCVA_downloaded column count: {count_cols_df_OHLCVA_downloaded}')
if count_symbols * 5 != count_cols_df_adjOHLCV:
  raise SystemExit(f'5 x symbol count: {5 * count_symbols} does not equal df_adjOHLCV column count: {count_cols_df_adjOHLCV}')
print(f'No error found, Column counts in df_close, df_OHLCVA, df_adjOHLCV matched symbol count')


# %%
print(f'count_symbols: {count_symbols}')
print(f'count_df_close_cols: {count_df_close_cols}')
print(f'count_cols_df_OHLCVA_downloaded: {count_cols_df_OHLCVA_downloaded}')
print(f'count_cols_df_adjOHLCV: {count_cols_df_adjOHLCV}')

# %%
close_vs_Yahoo(test_symbols, df_adjOHLCV)

# %%
_sym = 'NOC'
s_start, s_end = -252, -248
df_adjOHLCV[_sym].iloc[s_start:s_end]
df_OHLCVA_downloaded[_sym].iloc[s_start:s_end]

# %%
df_adjOHLCV.tail()

# %%
syms_def = ['NOC', 'HII', 'AVAV', 'LMT', 'CW', 'HEI']
_df = df_close[syms_def]
_df.tail()


# %%
syms_LNG = ['LNG', 'CHK', 'GLNG']
_df = df_close[syms_LNG]
_df.tail()

# %%
_df = df_close[symbols_add]
_df.tail()

# %%
# syms_def = ['NOC', 'HII', 'AVAV', 'LMT', 'CW', 'HEI']
s_start, s_end = -252, -247
_df = df_close[test_symbols].iloc[s_start:s_end]
_df


