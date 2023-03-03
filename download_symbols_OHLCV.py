import sys
import yfinance as yf

# path to myUtils
sys.path.insert(0, "C:/Users/ping/MyDrive/py_files/python/py379/")
# sys.path.append('/content/drive/MyDrive/py_files/python/py379/myUtils')
from myUtils import yf_download_AdjOHLCV, pickle_dump, pickle_load  # NOQA

verbose = False  # True prints more outputs

path_dir = "C:/Users/ping/MyDrive/stocks/MktCap2b_AUMtop1200/"
path_data_dump = path_dir + "VSCode_dump/"
path_symbols_file = path_dir + "source/"

##############################
filename_symbols = path_symbols_file + '2021_Top1200_MktCap_n_AUM.txt'
filename_pickled_df_OHLCV = 'df_OHLCV'  # pickled filename
# filename_symbols = path_symbols_file + "test_symbols_no_XOM.txt"
# filename_pickled_df_OHLCV = "df_test"  # pickled filename
##############################

index_symbol = "XOM"  # use Exxon's date index to re-index other symbols
df_XOM = yf.download(index_symbol)
# download OHLCV data for symbols in filename_symbols
df_OHLCV, symbols = yf_download_AdjOHLCV(filename_symbols, verbose=verbose)
# reindex to date index in df_XOM, weekend data are dropped
df_OHLCV = df_OHLCV.reindex(df_XOM.index, fill_value="NaN")

# pickle df_OHLCV and symbols
print(f"Full path to pickled df_OHLCV:  {path_data_dump}{filename_pickled_df_OHLCV}")
pickle_dump(df_OHLCV, path_data_dump, filename_pickled_df_OHLCV, verbose=verbose)
pickle_dump(symbols, path_data_dump, "symbols", verbose=verbose)
df_pickled = pickle_load(path_data_dump, filename_pickled_df_OHLCV, verbose=verbose)
mySyms = ["FTEC", "BCI", "BTC-USD", "ETH-USD"]
for sym in mySyms:
    print(f"df_pickled[{sym}]")
    print(df_pickled[sym], "\n")

symbols_pickled = pickle_load(path_data_dump, "symbols", verbose=verbose)
print(f"symbols_pickled:  {symbols_pickled}")
