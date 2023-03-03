import sys

# path to myUtils
sys.path.insert(0, "C:/Users/ping/MyDrive/py_files/python/py379/")
from myUtils import pickle_load  # NOQA

verbose = False  # True prints more outputs
# verbose = True  # True prints more outputs

# print OHLCV for these symbols
mySyms = ["FTEC", "GBTC", "BTC-USD", "ETH-USD", "XOM"]

path_dir = "C:/Users/ping/MyDrive/stocks/MktCap2b_AUMtop1200/"
path_data_dump = path_dir + "VSCode_dump/"
filename_pickle = "df_OHLCV"  # pickled filename

print(f"Full path to pickled df_OHLCV:  {path_data_dump}{filename_pickle}", "\n")  # NOQA
df_pickled = pickle_load(path_data_dump, filename_pickle, verbose=verbose)

for sym in mySyms:
    print(f"df_pickled[{sym}]")
    print(df_pickled[sym], "\n")
