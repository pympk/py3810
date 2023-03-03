from myUtils import yf_plot_symbols  # NOQA

# path_dir = "C:/Users/ping/MyDrive/stocks/MktCap2b_AUMtop1200/"
# path_data_dump = path_dir + "VSCode_dump/"

# filename_pickled_df_OHLCV = (
#     "df_OHLCV"  # pickled filename reindexed to NYSE dates
# )

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"
filename_pickled_df_OHLCV = (
    "df_OHLCV_clean"  # pickled filename reindexed to NYSE dates
)

symbols = ['SPY', 'FTEC']
# symbols = ["FTEC", "BCI", "BTC-USD", "ETH-USD"]

# date_end_limit = None
date_end_limit = "2023-02-28"
# iloc_offset = None  # number of days to plot
iloc_offset = 252  # number of days to plot
date_start_limit = None

caches_returned_by_plot = []
cache_returned_by_plot = yf_plot_symbols(
    symbols,
    path_data_dump,
    filename_pickled_df_OHLCV,
    date_start_limit,
    date_end_limit,
    iloc_offset,
)
caches_returned_by_plot.append(cache_returned_by_plot)
