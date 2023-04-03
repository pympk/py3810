import time


class Timer(object):
    """Timer to time code run time

        Ags:
            object('string'): name you want to give it

        Return:
            None

        Usage:
            with Timer('foo bar'):
                for i in range(100): # time 100 iteration
                    foo
                    bar
        Outputs:
            [foo bar]
            Elapsed: 1.666666 millisecond
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.clock()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        elapsed_ms = 1000*(time.clock() - self.tstart)  # time in millisecond
        print('Elapsed: %.6f millisecond' % elapsed_ms, '\n')


def get_file_names(path, ext):
    """Return file names that ends with the specified extension in a directory

    Search "path" directory, and return files names ends with "ext"

    Args:
        path(string): directory path, "C:/myfiles/"
        ext(string): file extension, ".csv", "" return all files

    Return:
        get_file_names(list): list of file names, e.g. ['AMZN.csv', 'QQQ.csv']
    """

    from os import listdir
    from os.path import isfile, join

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    get_file_names = [s for s in onlyfiles if s.endswith(ext)]
    # print("get_file_names = ", get_file_names)
    # print("type(get_file_names) = ", type(get_file_names), '\n')

    return (get_file_names)


def get_symbol_data_from_dir(dir_path, file_ext='.csv',
                             date_start=None, date_end=None, verbose=False):
    """Read symbol data from a directory and write them to dataframes

    Read symbol data, consists of date, open, high, low, close and volume,
    in .csv file format from a directory and write the data to a dictionary
    of dataframes.

    Dates are inclusive. None returns all data.

    Args:
        dir_path(string): directory path of the symbol data
        file_ext('string'): '.csv' default
        date_start('string'): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        date_end('string'): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        verbose(True or False): prints output if True, default False

    Return:
        dfs(dict}: dictionary of dataframes, e.g. dfs['QQQ']
        df_caches(dataframe): dataframe with columns: symbol, row_count,
            first_date, last_date, all_zeros_row_count; sorted by row_counts
        symbols[str]: list of symbols; ['AMZN', 'QQQ', 'SPY', ...]
    """

    import pandas as pd
    from myUtils import get_file_names

    # list of .csv files in the directory
    file_names = get_file_names(dir_path, file_ext)  # ['A.csv', 'QQQ.csv', ..]

    # column names of .csv files
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume']

    dfs = {}  # dictionary of dataframe for symbol data: date, open,...
    caches = []  # list for row counts, first date and last date for symbol
    symbols = []  # list for symbols

    for file_name in file_names:
        # use 1st column "date" as index
        df = pd.read_csv(dir_path + file_name, names=col_names,
                         parse_dates=True, index_col=0)

        symbol = file_name.replace(file_ext, '')  # strip .csv from file_name
        symbols.append(symbol)  # append symbol to list
        row_count = df.shape[0]
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        # collect symbol's all zeros rows into a df
        df_rows_of_zeros = df[(df.T == 0).all()]
        # count of rows with all zeros
        all_zeros_row_count = len(df_rows_of_zeros)

        # type(cache) = <class 'tuple'>
        # cache = ('XLK', 268, Timestamp('2017-06-01 00:00:00'),
        #          Timestamp('2018-06-22 00:00:00'))
        cache = (symbol, row_count, first_date, last_date,
                 all_zeros_row_count)  # tuple
        caches.append(cache)  # list

        df = df[date_start:date_end]  # select date range dates

        if df.empty:
            print('&'*20)
            print('df {} is empty:\n{}'.format(file_name, df))
            print('Available Data: df[{}:{}]:'
                  .format(first_date, last_date))
            print('Selected Data:  df[{}:{}]:'
                  .format(date_start, date_end))
            if last_date < date_start:
                print('Available Data ended {} *BEFORE* Selected Data start {}'
                      .format(last_date, date_start))
            print('&'*20, '\n')

        dfs.update({symbol: df})  # append symbol data to dictionary

    # sort caches by row_count in ascending order
    caches_sorted_by_row_count = sorted(caches, key=lambda tup: tup[1])

    # create df_caches dataframe
    df_caches = pd.DataFrame(
        columns=['symbol', 'row_count', 'first_date', 'last_date',
                 'all_zeros_row_count'])

    #  write to df_caches
    for index, (symbol, row_count, first_date, last_date, all_zeros_row_count)\
            in enumerate(caches_sorted_by_row_count):
        df_caches.loc[index] = \
            symbol, row_count, first_date, last_date, all_zeros_row_count

    if verbose:
        print('\n')
        print('+'*9, ' START OUTPUT: def get_symbol_data_from_dir ', '+'*9)
        print('\n')
        print('Number of ' + file_ext + ' files in directory = ',
              len(file_names), '\n')

        print('caches: ', '\n', caches, '\n')
        print('caches sorted by row_count:', '\n',
              caches_sorted_by_row_count, '\n')

        # print symbol dataframes stored in dictionary dfs
        print('='*30, ' dfs dataframes ', '='*30, '\n')
        for symbol in symbols:
            print('symbol: ', symbol)
            print(dfs[symbol].head(2), '\n')
            print(dfs[symbol].tail(2), '\n')
        print('='*76, '\n'*2)

        print('df_caches: ')
        print(df_caches.head(2), '\n')
        print(df_caches.tail(2), '\n')

        print('symbols: ', symbols, '\n')
        print('+'*9, ' END OUTPUT: def get_symbol_data_from_dir ',
              '+'*10, '\n'*2)

    return dfs, df_caches, symbols


def moving_window(arr, window):
    # reference: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    """Create moving-window of numpy.ndarray from a numpy.ndarray or pandas.Series

    Args:
        arr(float): a numpy.ndarray or pandas.Series
        window(int): width of moving-window

    Return:
        numpy.ndarray with shape [len(arr) - window + 1, window]
            where each row has elements of the moving_window,
            and number of elements in each row is the window size (i.e. columns)
    """

    import numpy as np

    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)

    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def symb_perf_stats(close):
    """Calculate performace metrics using symbol's closing price history
       v1 daily_return_std = np.std(daily_return, ddof=1) was
          daily_return_std = np.std(daily_return)

    Args:
        close(numpy array): symbol's closing price

    Return:
        cache(tuple): (period_yr, CAGR, CAGR_Std, CAGR_UI, daily_return_std,
            Std_UI, drawdown, ulcer_index, max_drawdown)
            period_yr(float) - closing price history in years
            CAGR(numpy.float64) - compounded annual growth rate
            CAGR_Std(numpy.float64) - CAGR / daily_return_std
            daily_return_std(numpy.float64) - std. deviation of daily returns
            Std_UI(numpy.float64) - daily_return_std / UlcerIndex
            drawdown(numpy.ndarray) - drawdown from peak, length same as close
            ulcer_index(numpy.float64) - Ulcer Index
            max_drawdown(numpy.float64) - maximum drawdown from peak
    """
    from ulcer_index import ulcer_index
    import numpy as np

    daily_return = close[1:] / close[:-1] - 1
    daily_return_std = np.std(daily_return, ddof=1)
    period_yr = len(close) / 252  # 252 trading days in a year
    # compounded annual growth rate
    CAGR = (close[-1] / close[0]) ** (1 / period_yr) - 1
    drawdown, ulcer_index, max_drawdown = ulcer_index(close)
    CAGR_UI = CAGR / ulcer_index  # original
    Std_UI = daily_return_std / ulcer_index  # larger is better
    CAGR_Std = CAGR / daily_return_std  # larger is better
    cache = (period_yr, CAGR, CAGR_Std, CAGR_UI, daily_return_std, Std_UI,
             drawdown, ulcer_index, max_drawdown)  # tuple

    return cache




def OBV_calc_(df, symbol, EMA_pd=10, tail_pd=5, norm_pd=30):
    """Calculate On-Balance-Volume(OBV), EMA of OBV,
       normalized EMA slope of the On-Balance-Volume (OBV).

    Args:
        df(dataframe): dataframe with date index
                       and columns open, high, low, close, volume
        symbol(str): symbol, e.g. 'QQQ'
        EMA_pd(int): exponential moving average period of OBV
        tail_pd(int): number of periods used to calculate avg_slope
        norm_pd(int): number of periods used to calculate avg_volume

    Return:
        OBV(numpy.ndarray): on-balance-volume
        OBV_EMA(numpy.ndarray): EMA of OBV
        normalized_OBV_EMA_slope(float): normalized average EMA slope,
          (average OBV_EMA difference for the last tail_pd days, default is 5) /
          (average volume for last norm_pd days, default is 30 days)
    """

    import talib as ta
    import numpy as np

    # close = df.close.values  # convert dataframe to numpy array
    close = df.close.values  # convert dataframe to numpy array
    # .astype(float) converts df.volume from integer to float for talib
    # volume = df.volume.values.astype(float)
    volume = df.volume.values.astype(float)
    OBV = ta.OBV(close, volume)
    OBV_EMA = ta.EMA(OBV, timeperiod=EMA_pd)  # calculate EMA of OBV
    # average OBV_EMA difference for the last tail_pd days, default is 5 days
    avg_slope = np.mean(np.diff(OBV_EMA[-(tail_pd+1):]))
    # average volume for last norm_pd days, default is 30 days
    avg_volume = np.mean(volume[-norm_pd:])
    # normalized avg_slope by avg_volume
    OBV_slope = avg_slope / avg_volume

    # # print statements for debug
    # print('\n')
    # print('*'*20)
    # print('symbol: ', symbol)
    # print('avg_slope: ', avg_slope)
    # print('avg_volume: ', avg_volume)
    # print('OBV_slope: ', OBV_slope)
    # print('*'*20, '\n')

    return OBV, OBV_EMA, OBV_slope


def OBV_calc(df, symbol, EMA_pd=10, tail_pd=5, norm_pd=30):
    """Calculate On-Balance-Volume(OBV), EMA of OBV,
       normalized EMA slope of the On-Balance-Volume (OBV).

    Args:
        df(dataframe): dataframe with date index
                       and columns open, high, low, close, volume
        symbol(str): symbol, e.g. 'QQQ'
        EMA_pd(int): exponential moving average period of OBV
        tail_pd(int): number of periods used to calculate avg_slope
        norm_pd(int): number of periods used to calculate avg_volume

    Return:
        OBV(numpy.ndarray): on-balance-volume
        OBV_EMA(numpy.ndarray): EMA of OBV
        normalized_OBV_EMA_slope(float): normalized average EMA slope,
          (average OBV_EMA difference for the last tail_pd days, default is 5) /
          (average volume for last norm_pd days, default is 30 days)
    """
    # df.Close.values was df.close.values
    # df.Volume.values.astype(float) was df.volume.values.astype(float)

    import talib as ta
    import numpy as np

    # close = df.close.values  # convert dataframe to numpy array
    close = df.Close.values  # convert dataframe to numpy array
    # .astype(float) converts df.volume from integer to float for talib
    # volume = df.volume.values.astype(float)
    volume = df.Volume.values.astype(float)
    OBV = ta.OBV(close, volume)
    OBV_EMA = ta.EMA(OBV, timeperiod=EMA_pd)  # calculate EMA of OBV
    # average OBV_EMA difference for the last tail_pd days, default is 5 days
    avg_slope = np.mean(np.diff(OBV_EMA[-(tail_pd+1):]))
    # average volume for last norm_pd days, default is 30 days
    avg_volume = np.mean(volume[-norm_pd:])
    # normalized avg_slope by avg_volume
    OBV_slope = avg_slope / avg_volume

    # # print statements for debug
    # print('\n')
    # print('*'*20)
    # print('symbol: ', symbol)
    # print('avg_slope: ', avg_slope)
    # print('avg_volume: ', avg_volume)
    # print('OBV_slope: ', OBV_slope)
    # print('*'*20, '\n')

    return OBV, OBV_EMA, OBV_slope






def nan_pad(arr_len, target_arr):
    """Pad target_arr with leading numpy.nan to length arr_len.

    Args:
        arr_len(int): length of return array
        target_arr(numpy.ndarray): array to be padded with leading numpy.nan

    Return:
        padded_target_arr(numpy.ndarray): target_arr padded to length arr_len
    """

    import numpy as np

    pad_len = arr_len - len(target_arr)
    pad_arr = np.empty(pad_len, dtype=float)
    pad_arr.fill(np.nan)
    padded_target_arr = np.concatenate((pad_arr, target_arr), axis=0)

    return padded_target_arr


def perf_stats_moving_window(arr_close, window=30):
    """Calculate performance metrics using a moving-window applied to
       the symbol's closing price history

    Args:
        close(numpy array): symbol's closing price
        window(int): moving-window size, default = 30

    Return:
        dataframe df_perf_stats padded with leading NaN to the same row count
            as arr_close

        columns of df_perf_stats are:
            CAGR(numpy.float64) - compounded annual growth rate
            CAGR_Std(numpy.float64) - CAGR / daily_return_std
            daily_return_std(numpy.float64) - std. deviation of daily returns
            Std_UI(numpy.float64) - daily_return_std / UlcerIndex
            ulcer_index(numpy.float64) - Ulcer Index
            max_drawdown(numpy.float64) - maximum drawdown from peak
    """

    import pandas as pd
    import numpy as np

    # apply moving-window to arr_close to create a N-dimensional array
    # shaped [len(arr_close)-window+1, window]
    close_moving_window = moving_window(arr_close, window)
    # list to store performance stats for each row of close_moving_window
    caches = []

    # calculate performance stats for each row of close_moving_window
    for close_arr in close_moving_window:
        # calculate performance stats
        period_yr, CAGR, CAGR_Std, CAGR_UI, daily_return_std,  Std_UI, \
            drawdown, ulcer_index, max_drawdown = symb_perf_stats(close_arr)
        # write perf. stats into cache as tuple
        #  each item written to cache is a number(float)
        #  did not cache drawdown since drawdown is an array of numbers
        cache = (period_yr, CAGR, CAGR_Std, CAGR_UI, daily_return_std,  Std_UI,
                 ulcer_index, max_drawdown)
        # write perf. stats of each row into caches as list
        caches.append(cache)  # list

    # create dataframe for performance stats
    df_perf_stats = pd.DataFrame(
        columns=['period_yr', 'CAGR', 'CAGR_Std', 'CAGR_UI',
                 'daily_return_std', 'Std_UI', 'ulcer_index', 'max_drawdown'])

    #  unpack caches to df_perf_stats
    for index, (period_yr, CAGR, CAGR_Std, CAGR_UI, daily_return_std, Std_UI,
                ulcer_index, max_drawdown) in enumerate(caches):
        df_perf_stats.loc[index] = period_yr, CAGR, CAGR_Std, CAGR_UI,\
            daily_return_std, Std_UI, ulcer_index, max_drawdown

    columns_selected = ['CAGR', 'CAGR_Std', 'CAGR_UI', 'daily_return_std',
                        'Std_UI', 'ulcer_index', 'max_drawdown']
    df_perf_stats = df_perf_stats[columns_selected]  # select needed columns

    # ++++ create df to pad df_perf_stats with leading NaN
    idx = list(range(len(arr_close) - len(df_perf_stats)))  # index for NaN df
    # create NaN df
    df_nan = pd.DataFrame(np.nan, index=idx, columns=columns_selected)
    # pad df_perf_stats with leading NaN
    df_perf_stats = pd.concat([df_nan, df_perf_stats], ignore_index=True)

    # error check: df_perf_stats and arr_close should have the same row count
    if len(arr_close) != len(df_perf_stats):
        raise Exception("len(close) should equal len(df_perf_stats), \n\
            len(close) was: {}, \n\
            len(df_perf_stats) was: {}"
                        .format(len(arr_close), len(df_perf_stats)))

    return df_perf_stats


def UI_MW(arr_drawdown, window=20):
    """Calculate Ulcer-Index on moving-window applied to
       arr_drawdown(numpy.ndarray)

    Args:
        arr_drawdown(numpy array): drawdown from function symb_perf_stats or
                                   from ulcer_index.py
        window(int): moving-window size, default = 20

    Return:
        UI_MW(numpy.ndarray): Ulcer-Index on moving-window
        applied to arr_drawdown
    """

    import numpy as np

    # apply moving window to arr_drawdown, result is numpy ndarray
    DD_MW = moving_window(arr_drawdown, window)
    DD_MW_square = np.square(DD_MW)
    sum_DD_MW_square = np.sum(DD_MW_square, axis=1)  # sum each row
    UI_MW = np.sqrt(sum_DD_MW_square / window)
    UI_MW = nan_pad(len(arr_drawdown), UI_MW)  # pad with leading NaN

    return UI_MW


def CAGR_MW(arr_close, window=20):
    """Calculate compounded-annual-growth-rate(CAGR) on moving-window applied to
       arr_close(numpy.ndarray)

    Args:
        arr_close(numpy array): symbol's daily price at close
        window(int): moving-window size, default = 20

    Return:
        CAGR_MW(numpy.ndarray): CAGR on moving-window applied to arr_close
    """

    import numpy as np

    period_year = window / 252  # period in years, 252 trading days per year
    # apply moving-window to arr_close
    close_MW = moving_window(arr_close, window)
    # first element from each row of the moving-window, type is 'list'
    close_first = [i[0] for i in close_MW]
    # last element from each row of the moving-window, type is 'list'
    close_last = [i[-1] for i in close_MW]
    close_first = np.asarray(close_first)  # convert to numpy array
    close_last = np.asarray(close_last)  # convert to numpy array
    arr_div = close_last / close_first  # numpy array
    # CAGR for each row of the moving-window
    CAGR_MW = (arr_div) ** (1 / period_year) - 1
    # pad with leading NaN to the same length as arr_close
    CAGR_MW = nan_pad(len(arr_close), CAGR_MW)

    return CAGR_MW


def df_symbols_close(path_symbols_data):
    """Return dataframe with closing prices for symbols in path_symbols_data

    Args:
        path_symbols_data(str): directory of symbols' .csv files

    Return:
        dataframe(pandas): dataframe with Date as an index and columns named
          after each symbol in the directory.  Each column is the daily
          closing price for that symbol.
    """

    from myUtils import get_symbol_data_from_dir
    import pandas as pd

    # load symbol data into dataframes
    dfs, df_row_count, symbols = \
        get_symbol_data_from_dir(path_symbols_data, file_ext='.csv',
                                 date_start=None, date_end=None)

    # create dataframe to store symbols' closing prices
    df_symbols_close = pd.DataFrame()

    for symbol in symbols:
        # get the symbol's date index and close column
        df_temp = dfs[symbol].close
        if symbol == symbols[0]:
            # first symbol, write close column to the empty dataframe
            df_symbols_close = pd.concat([df_temp], axis=1)
        else:
            # not the first symbol, append close column to dataframe
            df_symbols_close = pd.concat([df_symbols_close, df_temp], axis=1)

    # columns have the same name 'close', rename columns to match the symbols
    df_symbols_close.columns = symbols

    return df_symbols_close


def symbol_close(symbol_str, date_str, index_increment, df_symbols_close):
    """Return symbol's closing price for date specified by
       'date_str + index_increment'

    Args:
        symbol_str(str): symbol, e.g. 'QQQ'
        date_str(str): date in yyyy-mm-dd format, e.g. '2018-09-10',
          the date must be in the df_symbols_close.
        index_increment(int): date offset from date_str to get symbol's close
        df_symbols_close(dataframe): dataframe of symbols' closing prices
          returned by function df_symbols_close

    Return:
        symbol_close(float): symbol's close price on
          date = date_str + index_increment
    """

    # index location of date_str, returns keyError if location is not found
    idx = df_symbols_close.index.get_loc(date_str)
    new_idx = idx + index_increment
    # symbol's close price for the day after date_str
    symbol_close = df_symbols_close.iloc[new_idx][symbol_str]
    next_date = df_symbols_close.index[new_idx]

    # print('idx: ', idx)
    # print('date_str: ', date_str, type(date_str))
    # print('next_date: ', next_date, type(next_date))
    # print('symbol_close: ', symbol_close, type(symbol_close))

    return symbol_close, next_date


def directory_files(path, file_ext=''):
    """Return a list of file names in the path directory

    Args:
        path(str): diretory path, e.g. 'C:/Users/ping/Desktop/td_etf_data/'
        file_ext(str): file extension, default returns all files, e.g. '.csv'

    Return:
        only_files(list): list of file names, e.g. ['AAPL.csv', py_bonds.txt']
    """

    from os import listdir
    from os.path import isfile, join

    all_files = [f for f in listdir(path)
                 if isfile(join(path, f))]
    my_files = [file for file in all_files if file.endswith(file_ext)]

    return my_files


def get_df_symbols_close(path_df_symbols_close, df_file_name,
                         path_symbols_data):
    """Return dataframe of symbols' close. It reads data from file df_file_name
       in directory path_df_symbols_close, if the file is there. If the file is
       not there, it creates the file using symbols' .csv files in directory
       path_symbols_data

    Args:
        path_df_symbols_close(str): diretory path for df_file_name
        df_file_name(str): dataframe file name for symbols' close data
        path_symbols_data(str): directory for symbols' close .csv files

    Return:
        df(DataFrame): dataframe of symbols' close with Date (e.g. 2018-01-01),
          as index, symbols as column heading (e.g. 'AAPL', 'AMZN'), and
          symbol's close in each column
    """

    import pandas as pd
    import os
    import sys
    from myUtils import df_symbols_close, directory_files
    from datetime import datetime

    # path_file_name = path_symbols_data + df_file_name
    path_file_name = path_df_symbols_close + df_file_name

    warn_text_no_file = 'WARNING: No .csv file in {} \n\
Run YLoader to download symbol data to directory.'.format(path_symbols_data)

    # test if dataframe df_symbols_close.txt exists in directory
    if not os.path.isfile(path_df_symbols_close + df_file_name):
        # df_symbols_close.txt does not exist
        print('Dataframe "{}" does not exist \n'.format(path_file_name))

        # get symbols' .csv file names from path_symbols_data
        # my_files = directory_files(path=path_symbols_data, file_ext='.csv')
        # print('my_files: ', my_files, type(my_files), '\n')
        symbol_files = directory_files(path=path_symbols_data, file_ext='.csv')
        print('symbol_files:\n{}\n'.format(symbol_files))
        print('type(symbol_files):  {}\n'.format(type(symbol_files)))

        if not symbol_files:  # empty list returns False
            # no .csv files in path_symbols_data directory
            print('='*81)
            print(warn_text_no_file)
            print('='*81, '\n')
            sys.exit()  # no symbol's .csv file in directory, exit script
        else:  # path_symbols_data directory has symbol's .csv file
            # write symbols' close data to dataframe
            df = df_symbols_close(path_symbols_data=path_symbols_data)
            # change date index from 2018-09-04 00:00:00 to 2018-09-04
            #   dtype also change from datetime64[ns] to object
            df.index = df.index.strftime("%Y-%m-%d")
            df.index.name = 'Date'
            df.to_csv(path_df_symbols_close + df_file_name)  # store file
            print("Wrote symbols' close dataframe to: \n{} \n"
                  .format(path_df_symbols_close + df_file_name))
    else:  # dataframe df_symbols_close.txt exists in directory
        # getmtime returns file's modification date as float
        file_date = os.path.getmtime(path_df_symbols_close + df_file_name)
        # convert date from float to string
        file_date = datetime.fromtimestamp(file_date).strftime('%Y-%m-%d')
        # get current date
        current_date = datetime.today().strftime('%Y-%m-%d')
        print('Dataframe "{}" exists \n'.format(path_file_name))

        if file_date == current_date:  # file_date same as today's date
            print('Dataframe "{}" date is the same as current date {} \n'
                  .format(path_file_name, file_date))
        else:  # file_date not the same as today's date
            print('Dataframe "{}", modified on {}, is NOT the current date {}. \
Overwrite the existing dataframe. \n'.format(path_file_name,
                                             file_date, current_date))
#            print('Dataframe "{}" date is NOT the same as current date {}. \
# Overwrite the existing dataframe. \n'.format(path_file_name, file_date))
            # overwrite dataframe df_symbols_close.txt
            df = df_symbols_close(path_symbols_data=path_symbols_data)
            # change date index from 2018-09-04 00:00:00 to 2018-09-04
            #   dtype also change from datetime64[ns] to object
            df.index = df.index.strftime("%Y-%m-%d")
            df.index.name = 'Date'
            df.to_csv(path_df_symbols_close + df_file_name)  # store file
            print("Wrote symbols' close dataframe to: \n{} \n"
                  .format(path_df_symbols_close + df_file_name))

        # dataframe df_symbols_close.txt exists, read it into df
        df = pd.read_csv(path_df_symbols_close + df_file_name, index_col=0)

    return df


def reindex_df(index_sample, df):
    """Return dataframe df re-indexed to index_sample. If new rows are added,
       they will be filled with NaNs.

    Args:
        index_sample(pandas index): pandas index example
        df(DataFrame): dataframe to be re-indexed

    Return:
        df_reindex(DataFrame): dataframe, df, re-indexed to index_sample
    """

    # select df index that matches index_sample
    valid_index = df.index.intersection(index_sample)
    # reindex df to index_sample
    df_reindex = df.loc[valid_index].reindex(index_sample)

    return df_reindex


def remove_NaN_rows(df, max_NaN_per_row, verbose=True):
    """Removes rows with NaN count more than the max_NaN_per_row from the
       dataframe df.

    Args:
        df(dataframe): dataframe
        max_NaN_per_row(float): maximum allowable NaN count per row

    Return:
        df(DataFrame): dataframe with NaN count of each row less than
        the max_NaN_per_row
    """

    print('\n{}'.format('='*78))
    print('+ def remove_NaN_rows(df, max_NaN_per_row, verbose=True)\n')
    print('max_NaN_per_row: ', max_NaN_per_row)
    print('df.shape before NaN row removal: ', df.shape, '\n')
    # df_nan_rows is a pd.series with a count of NaN for each row
    #  isnull turns df NaN element into True
    #  sum(axis=1) sums the True in each row
    df_nan_rows = df.isnull().sum(axis=1)  # pandas core series
    # drop rows with NaN count more than ax_NaN_per_row
    df_nan_rows = df_nan_rows[df_nan_rows > max_NaN_per_row]

    if verbose is True:
        print('These indexes and symbols have more than {} NaN per row. \n\
They were removed from dataframe.\n'.format(max_NaN_per_row))
        # print('df_nan_rows: ')
        # print(df_nan_rows, '\n')
        print('These "df" index rows have NaN:')
        for index, val in df_nan_rows.iteritems():
            print('index {} symbol {} has {} NaN'
                  .format(index, df.symbol[index], val))
        print('\n')

    # remove NaN rows from df using the index from df_nan_rows
    df = df.drop(df_nan_rows.index.values, axis=0)
    # print('\n')
    print('df.shape after NaN row removal : ', df.shape)
    # print('='*82, '\n')
    print('- def remove_NaN_rows(df, max_NaN_per_row, verbose=True)')
    print('{}\n'.format('-'*78))

    return df


def remove_NaN_columns(df, max_NaN_per_column, verbose=True):
    """Removes columns with NaN count more than the max_NaN_per_column from the
       dataframe df.

    Args:
        df(dataframe): dataframe
        max_NaN_per_column(float): maximum allowable NaN count per column
        verbose(bool): if True, print index and column name

    Return:
        df(DataFrame): dataframe with NaN count of each column less than
        the max_NaN_per_column
    """

    print('='*30, ' remove_NaN_columns ', '='*30)
    print('max_NaN_per_column: ', max_NaN_per_column, '\n')
    print('df.shape before NaN column removal: ', df.shape, '\n')
    # df_nan_columns is a pd.series with a count of NaN for each column
    #  isnull turns df NaN element into True
    #  sum(axis=0) sums the True in each column
    df_nan_columns = df.isnull().sum(axis=0)  # pandas core series
    # drop columns with NaN count more than ax_NaN_per_column
    df_nan_columns = df_nan_columns[df_nan_columns > max_NaN_per_column]

    if verbose is True:
        print('These columns have more than {} NaN per column. \n\
They were removed from dataframe.'.format(max_NaN_per_column))
        for index, val in df_nan_columns.iteritems():
            print('column {} has {} NaN'
                  .format(index, val))

    # remove NaN rows from df using the index from df_nan_column
    df = df.drop(df_nan_columns.index.values, axis=1)
    print('\n')
    print('df.shape after NaN column removal: ', df.shape)
    print('='*81, '\n')

    return df


def print_list_in_groups(my_list, group_size):
    """Print elements in my_list in groups of group_size.

    Args:
        my_list(list or ndarray): list to be printed in groups
        group_size(int) = my_list's element group size

    Return:
        None
    """

    print('\n{}'.format('='*78))
    print('def print_list_in_groups(my_list, group_size)')
    print('group size = {}'.format(group_size))
    my_list = list(my_list)  # if my_list is an ndarray, convert it to a list
    for i in range(0, len(my_list), group_size):
        print('{:<3d} : {}'.format(i, my_list[i:i+group_size]))
    print('{}\n'.format('-'*78))


def list_dump(list_to_dump, path_list_dump, filename, verbose=False):
    """Write list_to_dump to a text file to path_list_dump with the file name
       filename.

    Args:
        list_to_dump(str): list to write to file
        path_list_dump(str): path to write to file
        filename(str): text file name, e.g. symbols.txt, symbols.csv

    Return:
        None
    """

    with open(path_list_dump + filename, 'w') as f:
        for item in list_to_dump:
            f.write("%s\n" % item)
    if verbose:
        print('\n{}'.format('='*78))
        print('+ def list_dump(list_to_dump, path_list_dump, filename)\n')
        print('Wrote list to {}{}'
              .format(path_list_dump, filename))
        print('- def list_dump(list_to_dump, path_list_dump, filename)\n')
        print('{}\n'.format('-'*78))


def pickle_dump(file_to_pickle, path_pickle_dump, filename_pickle,
                verbose=False):
    """Pickle file_to_pickle and write it to path_pickcle_dump with the filename
       filename_pickle.

    Args:
        file_to_pickle(python object): file to be pickle
        path_pickcle_dump(str): path to write the pickle file
        filename_pickle(str): pickle filename

    Return:
        None
    """

    import pickle

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def pickle_dump(file_to_pickle, path_pickle_dump, filename_pickle)\n')  # NOQA
    outfile = open(path_pickle_dump + filename_pickle, 'wb')
    pickle.dump(file_to_pickle, outfile)
    outfile.close()
    if verbose:    
        print('Wrote pickled file to {}{}'
              .format(path_pickle_dump, filename_pickle))
        print('- def pickle_dump(file_to_pickle, path_pickle_dump, filename_pickle)')  # NOQA
        print('{}\n'.format('-'*78))


def pickle_load(path_pickle_dump, filename_pickle, verbose=False):
    """Un-pickle filename_pickle from path_pickle_dump and return the result.

    Args:
        path_pickle_dump(str): path to retrieve pickle file
        filename_pickle(str): pickle filename

    Return:
        file_unpickle: un-pickle file
    """

    import pickle

    infile = open(path_pickle_dump + filename_pickle, 'rb')
    file_unpickle = pickle.load(infile)
    infile.close()

    if verbose:
        print('\n{}'.format('='*78))
        print('def pickle_load(path_pickle_dump, filename_pickle)')
        print('Read pickle file from:\n{}{}'
            .format(path_pickle_dump, filename_pickle))
        print('{}\n'.format('-'*78))

    return file_unpickle


def list_sort(my_list, verbose=False):
    """Given a list with duplicate elements, the function returns a list of
       unique elements sorted in descending order of duplicate element counts.

    Args:
        my_list(list): list with duplicate elements
        verbose(bool): enables printing of more outputs

    Return:
        list_keys(list): list of unique elements in descending order of
            duplicate element counts
        list_counts(list): list of integers corresponding to the duplicate
            element counts in list_keys
        list_tuples_sorted(list): sorted list of tuples, each tuple has the
            unique symbol and the count of that symbol.
    """

    from collections import Counter

    if verbose:
        print('\n{}'.format('='*78))
        print('def list_sort(my_list)\n')
    if verbose:
        print('my_list:\n{}\n'.format(my_list))

    list_keys = []
    list_counts = []
    # Counter(my_list) removes duplicates from my_list and creates
    #   a unique list of tuples (e.g. [('VWOB', 1), ('VCSH', 3), ('BNDX', 3)]).
    #   most_common() sorts the unique list in desending order of frequency:
    #   (e.g. [('VCSH', 3), ('BNDX', 3), ('VWOB', 1)])
    list_tuples_sorted = Counter(my_list).most_common()
    if verbose:
        print('{:<10s} : {:<5s}'.format('Keys', 'Counts'))
    for i in list_tuples_sorted:
        if verbose:
            print('{:<10s} : {:<5d}'.format(i[0], i[1]))
        list_keys.append(i[0])
        list_counts.append(i[1])
    if verbose:
        print('{}\n'.format('-'*78))

    return list_keys, list_counts, list_tuples_sorted


def replace_rows_of_zeros(df, symbol, verbose=False):
    """Replace dataframe's rows-with-all-zeros with values
       from the previous rows

    Args:
        df(dataframe): dataframe of symbol's date, open, high, low, close
           and volume
        symbol(str): symbol name (e.g. 'AMZN')

    Return:
        df(dataframe): dataframe of symbol's date, open, high, low, close
           and volume with all-zeros-rows replaced by values from previous rows
    """

    print('\n{}'.format('='*78))
    print('def replace_rows_of_zeros(df, symbol)\n')

    # collect symbol's all zeros rows into a df
    df_rows_of_zeros = df[(df.T == 0).all()]
    # count of rows with all zeros
    count_zero_rows = len(df_rows_of_zeros)

    if count_zero_rows > 0:
        index_mask = df_rows_of_zeros.index[::]  # index of all-zeros-rows
        if verbose:
            print('\n{} {} {}'
                  .format('='*23, 'def replace_rows_of_zeros(df)', '='*23))
            print('[symbol], [index_mask], [count_zero_rows]:\n[{}], [{}], [{}]\n'  # NOQA  
                  .format(symbol, index_mask, count_zero_rows))
            print('df.loc[index_mask]:\n{}\n'.format(df.loc[index_mask]))

        for idx in index_mask:
            loc = df.index.get_loc(idx)
            prev_loc = loc - 1
            df.iloc[loc] = df.iloc[prev_loc]
            if verbose:
                print('idx: {}: '.format(idx))
                print('loc: {}'.format(loc))
                print('df.iloc[prev_loc]:\n{}\n'.format(df.iloc[prev_loc]))

    print('{}\n'.format('-'*78))

    return df


def date_newest_in_range(date_index, date_range_end):
    """Given date range from date_index[0] to date_range_end,
       the function return the newest date within the given range.

    Args:
        date_index(DatetimeIndex): datetime index, e.g. ['2019-11-08', ...]
        date_range_end(str): end-of-range date in format 'yyyy-mm-dd',
            e.g. '2018-12-31', start-of-range is date_index[0]

    Return:
        newest_date(pandas Timestamp): newest index date within the given date
            range (e.g 2019-11-15 00:00:00)
        newest_date_iloc(int): index location of newest_date (e.g. 14569)
        date_range_end_in_index(bool): is date_range_end in the date index
            (e.g. True/False)
    """

    from datetime import datetime as dt

    print('\n{}'.format('='*78))
    print('def date_newest_in_range(date_index, date_range_end)\n')

    date_range_end =\
        dt.strptime(date_range_end, "%Y-%m-%d")  # convert string to datetime

    if date_range_end < date_index[0]:
        error_msg = 'date_end_range {} MUST BE NEWER than date_index[0] {}: '
        raise ValueError(error_msg.format(date_range_end, date_index[0]))
    else:
        # code using list-comprehension
        newest_date = \
            max(date for date in date_index if date <= date_range_end)
        # # code using for-loop
        # newer_date = []
        # for date in date_index:
        #     if date <= date_range_end:
        #         newer_date.append(date)
        # newest_date = max(newer_date)

        newest_date_iloc = \
            [i for i, x in enumerate(date_index) if x == newest_date]
        newest_date_iloc = newest_date_iloc[0]  # convert list to int

        date_range_end_in_index = date_range_end in date_index

    print('{}\n'.format('-'*78))

    return newest_date, newest_date_iloc, date_range_end_in_index


def valid_date(date_text, verbose=False):
    """Check date_text string for a valid date and is in 'yyyy-mm-dd' format.
    Args:
        date_text(str): date in format 'yyyy-mm-dd'

    Return:
        date_datetime(datetime): date in datetime format
            (e.g 2019-11-15 00:00:00)
    """

    import datetime

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def valid_date(date_text)\n')

    try:
        date_datetime = datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        error_msg = \
            '{} is an invalid date or not in this format: yyyy-mm-dd'
        raise ValueError(error_msg.format(date_text))
    if verbose:
        print('- def valid_date(date_text)')
        print('{}\n'.format('-'*78))
    return date_datetime


def valid_start_end_dates(date_start, date_end, verbose=False):
    """Check for: 1. dates are either in format 'yyyy-mm-dd' or None
                  2. dates are valid dates
                  3. date_start older then date_end
    Args:
        date_start(str): date in format 'yyyy-mm-dd' or None
        date_end(str): date in format 'yyyy-mm-dd' or None

    Return: Not applicable
    """

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def valid_start_end_dates(date_start, date_end)\n')

    both_are_None = \
        date_start is None and date_end is None
    end_is_None = \
        date_start is not None and date_end is None
    start_is_None = \
        date_start is None and date_end is not None
    both_are_not_None = \
        date_start is not None and date_end is not None

    if both_are_None:
        if verbose:        
            print('\nValid Dates: Start and End dates are None')
    elif end_is_None:
        valid_date(date_start)  # throw error if date is invalid
        if verbose:
            print('\nValid Dates: End date is None')
    elif start_is_None:
        valid_date(date_end)  # throw error if date is invalid
        if verbose:
            print('\nValid Dates: Start Date is None')
    elif both_are_not_None:
        start_date = valid_date(date_start)  # throw error if date is invalid
        end_date = valid_date(date_end)  # throw error if date is invalid
        if start_date >= end_date:  # invalid, start-date newer than end-date
            error_msg_start_gtr_end = \
                'Invalid Dates: date_start {} MUST be older than date_end {}'
            raise (ValueError(error_msg_start_gtr_end
                   .format(start_date, end_date)))
        if verbose:
            print('\nValid Dates: Start and End dates are not None')

    if verbose:
        print('- def valid_start_end_dates(date_start, date_end)')
        print('{}\n'.format('-'*78))


def dates_adjacent_pivot(date_index, date_pivot, verbose=False):
    """The function checks the pivot date is within the dates in the index.
       It returns an older or equal index date that is adjacent to the pivot
       date, and a younger or equal index date that is adjacent to the pivot
       date. It returns the index locations of the older and younger dates.
       It also returns, True/False, if the pivot date is in the index.

    Args:
        date_index(DatetimeIndex): datetime index, e.g. ['2019-11-08', ...]
        date_pivot(str): date in format 'yyyy-mm-dd'

    Return:
        date_pivot_or_older(pandas Timestamp): older or equal index date that
            is adjacent to date_pivot
        date_pivot_or_younger(pandas Timestamp): younger or equal index date
            that is adjacent to date_pivot
        iloc_date_pivot_or_older(int): index location of date_pivot_or_older
            (e.g. 14569)
        iloc_date_pivot_or_younger(int): index location of
            date_pivot_or_younger (e.g. 14569)
        pivot_date_in_index(bool): is date_pivot in the date index
            (e.g. True/False)
    """

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def dates_adjacent_pivot(date_index, date_pivot)\n')

    from datetime import datetime as my_dt

    date_pivot =\
        my_dt.strptime(date_pivot, "%Y-%m-%d")  # convert string to datetime

    # An older date has a smaller value than a younger date
    #  date_pivot is older than the first index date
    if date_pivot < date_index[0]:
        error_msg = \
            'date_pivot {} MUST BE YOUNGER than the first index date {}'
        raise ValueError(error_msg.format(date_pivot, date_index[0]))
    # date_pivot is younger than the last index date
    elif date_pivot > date_index[-1]:
        error_msg = 'date_pivot {} MUST BE OLDER than the last index date {}'
        raise ValueError(error_msg.format(date_pivot, date_index[-1]))
    else:
        # code using list-comprehension
        date_pivot_or_older = \
            max(date for date in date_index if date <= date_pivot)
        # # code using for-loop
        # younger_date = []
        # for date in date_index:
        #     if date <= date_pivot:
        #         younger_date.append(date)
        # date_pivot_or_older = max(younger_date)

        date_pivot_or_younger = \
            min(date for date in date_index if date >= date_pivot)

        iloc_date_pivot_or_older = \
            [i for i, x in enumerate(date_index) if x == date_pivot_or_older]
        iloc_date_pivot_or_younger = \
            [i for i, x in enumerate(date_index) if x == date_pivot_or_younger]
        # convert list to int
        iloc_date_pivot_or_older = iloc_date_pivot_or_older[0]
        iloc_date_pivot_or_younger = iloc_date_pivot_or_younger[0]

        pivot_date_in_index = date_pivot in date_index  # True/False

    if verbose:
        print('- def dates_adjacent_pivot(date_index, date_pivot)')
        print('{}\n'.format('-'*78))

    return \
        date_pivot_or_older, \
        date_pivot_or_younger, \
        iloc_date_pivot_or_older, \
        iloc_date_pivot_or_younger, \
        pivot_date_in_index


def dates_within_limits(index, date_start_limit=None, date_end_limit=None, iloc_offset=None, verbose=False):
    # TODO: rewrite code using pandas.date_range
    """The function selects dates in the index that are within the 
       date_start_limit and the date_end_limit, and returns the oldest date in
       the selection as date_start, and the newest in the selection as
       date_end.  date_start and date_end will be at, or within, these limits.

       Order of hierarchy to determine Start and End dates:
       1. given date if it is not None
       2. iloc_offset if it is not None

       start = date_start_limit
       end = date_end_limit
       date_start_limit sets the upper (i.e. oldest) limit.
       date_end_limit sets the lower (i.e. newest) limit.
       date_start is the oldest index date that is within the limits.
       date_end is the newest index date that is within the limits.

       START   END    ILOC-OFFSET  Return->  START DATE             END DATE
       ---------------------------------------------------------------------------
       start   end	  iloc_offset	         date_start	            date_end
       start   end	  none                   date_start	            date_end
       start   none   iloc_offset	         date_start	            date_start+iloc_offset  # NOQA
       start   none   none                   date_start	            index[-1]
       none	   end	  iloc_offset	         date_end-iloc_offset   date_end
       none	   end	  none                   index[0]	            date_end
       none	   none   iloc_offset	         index[0]	            index[-1]
       none	   none   none                   index[0]	            index[-1]

    Args:
        index(DatetimeIndex): datetime index, e.g. ['2019-11-08', ...]
        date_start_limit(str): limit on the start date in format 'yyyy-mm-dd'
        date_end_limit(str): limit on the end date in format 'yyyy-mm-dd'
        iloc_offset(int): index offset
        verbose(bool): True/False, print output

    Return:
        date_start(pandas Timestamp): an index date that is newer or equal to
            date_start_limit
        date_end(pandas Timestamp): an index date that is older or equal to
            date_end_limit
        iloc_date_start(int): index location of date_start
        iloc_date_end(int): index location of date_end
    """

    # import pandas as pd
    from myUtils import valid_start_end_dates, dates_adjacent_pivot

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def dates_within_limits(index, date_start_limit, date_end_limit, iloc_offset)\n')  # NOQA

    print_format1 = '{:<30s}  {}'  # print format

    # validate iloc_offset
    if iloc_offset is not None and iloc_offset < 0:
        error_msg_iloc_offset = \
            'iloc_offset = {}, It must be a positive interger or "None".\n'
        raise ValueError(error_msg_iloc_offset
                         .format(iloc_offset))

    # check for: 1. valid date, None is also valid
    #            2. in 'yyyy-mm-dd' format
    #            3. start date is older than end date
    valid_start_end_dates(date_start_limit, date_end_limit)
    if verbose:
        print('\nindex: {}\n'.format(index))
        print(print_format1.format('date_start_limit', date_start_limit))
        print(print_format1.format('date_end_limit', date_end_limit))
        print(print_format1.format('iloc_offset', iloc_offset))

    # region: tags for inputs
    start_end_iloc = \
        date_start_limit is not None and \
        date_end_limit is not None and \
        iloc_offset is not None
    start_end_none = \
        date_start_limit is not None and \
        date_end_limit is not None and \
        iloc_offset is None
    start_none_iloc = \
        date_start_limit is not None and \
        date_end_limit is None and \
        iloc_offset is not None
    start_none_none = \
        date_start_limit is not None and \
        date_end_limit is None and \
        iloc_offset is None
    none_end_iloc = \
        date_start_limit is None and \
        date_end_limit is not None and \
        iloc_offset is not None
    none_end_none = \
        date_start_limit is None and \
        date_end_limit is not None and \
        iloc_offset is None
    none_none_iloc = \
        date_start_limit is None and \
        date_end_limit is None and \
        iloc_offset is not None
    none_none_none = \
        date_start_limit is None and \
        date_end_limit is None and \
        iloc_offset is None
    # endregion

    if start_end_iloc or start_end_none:
        # return: date_start <= date_start_limit, date_end => date_end_limit
        date_pivot_or_older, date_pivot_or_younger, iloc_date_pivot_or_older, \
            iloc_date_pivot_or_younger, pivot_date_in_index = \
            dates_adjacent_pivot(date_index=index, date_pivot=date_start_limit)
        date_start = date_pivot_or_younger
        iloc_date_start = iloc_date_pivot_or_younger
        date_start_in_index = pivot_date_in_index

        date_pivot_or_older, date_pivot_or_younger, iloc_date_pivot_or_older, \
            iloc_date_pivot_or_younger, pivot_date_in_index = \
            dates_adjacent_pivot(date_index=index, date_pivot=date_end_limit)
        iloc_date_end = iloc_date_pivot_or_older
        date_end = date_pivot_or_older
        date_end_in_index = pivot_date_in_index

        message_Truth_Table_logic = \
            'Since input has Start and End dates, iloc_offset is not used'

    elif start_none_iloc:
        # return: date_start <= date_start_limit
        #  date_end = date_start + iloc_offset
        date_pivot_or_older, date_pivot_or_younger, iloc_date_pivot_or_older, \
            iloc_date_pivot_or_younger, pivot_date_in_index = \
            dates_adjacent_pivot(date_index=index, date_pivot=date_start_limit)
        date_start = date_pivot_or_younger
        iloc_date_start = iloc_date_pivot_or_younger
        date_start_in_index = pivot_date_in_index

        iloc_date_end = iloc_date_start + iloc_offset

        if iloc_date_end > len(index) - 1:
            error_msg_iloc_overflow = \
                'iloc_date_start + iloc_offset > len(index) - 1\n\
iloc_date_start:  {}\n\
iloc_offset:      {}\n\
len(index) - 1:   {}\n'
            raise ValueError(error_msg_iloc_overflow
                             .format(iloc_date_start, iloc_date_end,
                                     iloc_offset))

        message_Truth_Table_logic = \
            'date_start_limit and iloc_offset are used, \
date_end_limit is not used\n'

        date_end = index[iloc_date_end]
        date_end_in_index = True

    elif start_none_none:
        # return: date_start <= date_start_limit, date_end = index[-1]
        date_pivot_or_older, date_pivot_or_younger, iloc_date_pivot_or_older, \
            iloc_date_pivot_or_younger, pivot_date_in_index = \
            dates_adjacent_pivot(date_index=index, date_pivot=date_start_limit)
        date_start = date_pivot_or_younger
        iloc_date_start = iloc_date_pivot_or_younger
        date_start_in_index = pivot_date_in_index

        message_Truth_Table_logic = \
            'date_start_limit, date_end_limit and iloc_offset are not used\n'

        iloc_date_end = len(index) - 1
        date_end = index[iloc_date_end]
        date_end_in_index = True

    elif none_end_iloc:
        # return: date_start = date_end - iloc_offset,
        #  date_end => date_end_limit
        date_pivot_or_older, date_pivot_or_younger, iloc_date_pivot_or_older, \
            iloc_date_pivot_or_younger, pivot_date_in_index = \
            dates_adjacent_pivot(date_index=index, date_pivot=date_end_limit)
        iloc_date_end = iloc_date_pivot_or_older
        date_end = date_pivot_or_older
        date_end_in_index = pivot_date_in_index

        iloc_date_start = iloc_date_end - iloc_offset
        if iloc_date_start < 0:
            error_msg_iloc_start_less_than_0 = \
                'iloc_date_start must be > 0\n\
iloc_date_start = iloc_date_end - iloc_offset\n\
iloc_date_end:    {}\n\
iloc_offset:      {}\n\
iloc_date_start:  {}\n\
Change date_end to a later date, or use a smaller iloc_offset\n'
            raise ValueError(error_msg_iloc_start_less_than_0
                             .format(iloc_date_end, iloc_offset,
                                     iloc_date_start))
        date_start = index[iloc_date_start]
        date_start_in_index = True

        message_Truth_Table_logic = \
            'date_end_limit and iloc_offset are used, date_start_limit is not used\n'  # NOQA

    elif none_end_none:
        # return: date_start = index[0], date_end => date_end_limit
        date_pivot_or_older, date_pivot_or_younger, iloc_date_pivot_or_older, \
            iloc_date_pivot_or_younger, pivot_date_in_index = \
            dates_adjacent_pivot(date_index=index, date_pivot=date_end_limit)
        iloc_date_end = iloc_date_pivot_or_older
        date_end = date_pivot_or_older
        date_end_in_index = pivot_date_in_index

        iloc_date_start = 0
        date_start = index[iloc_date_start]
        date_start_in_index = True

        message_Truth_Table_logic = \
            'date_end_limit is used, date_start_limit and iloc_offset are not used\n'  # NOQA

    elif none_none_iloc or none_none_none:
        # return: date_start = index[0], date_end = index[-1]
        iloc_date_start = 0
        date_start = index[iloc_date_start]
        date_start_in_index = True

        iloc_date_end = len(index) - 1
        date_end = index[iloc_date_end]
        date_end_in_index = True

        message_Truth_Table_logic = \
            'date_start_limit, date_end_limit and iloc_offset are not used\n'

    # number of index dates between date_start and date_end
    count_index_date = iloc_date_end - iloc_date_start

    # region: print verbose statements
    if verbose:
        message_inputs = \
            'Input has:\n\
    date_start_limit                {}\n\
    date_end_limit                  {}\n\
    iloc_offset                     {}\n'\
    .format(date_start_limit, date_end_limit, iloc_offset)

        # print('\n' + message_inputs + message_Truth_Table_logic)
        print('\n' + message_inputs + '\n' + message_Truth_Table_logic)
        print(print_format1.format('date_start', date_start))
        # sanity check, date_start is the same as index[iloc_date_start]
        print(print_format1
              .format('index[iloc_date_start]', index[iloc_date_start]))
        print(print_format1.format('iloc_date_start', iloc_date_start))
        print(print_format1.format('date_start_in_index', date_start_in_index))
        print('')
        print(print_format1.format('date_end', date_end))
        # sanity check, date_end is the same as index[iloc_date_end]
        print(print_format1
              .format('index[iloc_date_end]', index[iloc_date_end]))
        print(print_format1.format('iloc_date_end', iloc_date_end))
        print(print_format1.format('date_end_in_index', date_end_in_index))
        print('')
        print(print_format1
              .format('count_index_date(0-based)', count_index_date))
    # endregion

    if verbose:
        print('- def dates_within_limits(index, date_start_limit, date_end_limit, iloc_offset)')  # NOQA
        print('{}\n'.format('-'*78))

    return date_start, date_end, iloc_date_start, iloc_date_end


def get_pooled_symbol_data(file_symbol_names, path_pooled_symbol_data,
                           file_ext='.csv', date_start=None, date_end=None,
                           verbose=False):
    """Read symbol names from file_symbol_names, then load symbol data from
    path_pooled_symbol_data and write them to dataframes.  Symbol data were
    downloaded to directory path_pooled_symbol_data using YLoader.

    Dates are inclusive. None returns all data.

    Args:
        file_symbol_names(str): .txt file of the symbols, including path
        path_pooled_symbol_data(str): directory path of the symbol data
        file_ext('str'): symbol data file extension, '.csv' default
        date_start('str'): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        date_end('str'): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        verbose(bool): prints output if True, default False

    Return:
        dfs(dict}: dictionary of dataframes, e.g. dfs['QQQ']
        df_caches(dataframe): dataframe with columns: symbol, row_count,
            first_date, last_date, all_zeros_row_count; sorted by row_counts
        symbols[str]: list of symbols; ['AMZN', 'QQQ', 'SPY', ...]
    """

    import pandas as pd

    print('+ def get_pooled_symbol_data(file_symbol_names, \
path_pooled_symbol_data, file_ext=".csv", date_start=None, date_end=None, \
verbose=False)\n')

    # read lines from .txt file to list
    with open(file_symbol_names, 'r') as f:
        # exclude the newlines (\n)
        symbol_names = f.read().splitlines()

    # column names of .csv files
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume']

    dfs = {}  # dictionary of dataframe for symbol data: date, open,...
    caches = []  # list for row counts, first date and last date for symbol
    symbols = []  # list for symbols

    for symbol in symbol_names:
        # use 1st column "date" as index
        df = pd.read_csv(path_pooled_symbol_data + symbol + file_ext,
                         names=col_names, parse_dates=True, index_col=0)

        symbols.append(symbol)  # append symbol to list
        row_count = df.shape[0]
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        # collect symbol's all zeros rows into a df
        df_rows_of_zeros = df[(df.T == 0).all()]
        # count of rows with all zeros
        all_zeros_row_count = len(df_rows_of_zeros)

        # type(cache) = <class 'tuple'>
        # cache = ('XLK', 268, Timestamp('2017-06-01 00:00:00'),
        #          Timestamp('2018-06-22 00:00:00'))
        cache = (symbol, row_count, first_date, last_date,
                 all_zeros_row_count)  # tuple
        caches.append(cache)  # list

        df = df[date_start:date_end]  # select date range dates

        if df.empty:
            print('&'*20)
            print('df {} is empty:\n{}'.format(symbol, df))
            print('Available Data: df[{}:{}]:'
                  .format(first_date, last_date))
            print('Selected Data:  df[{}:{}]:'
                  .format(date_start, date_end))
            if last_date < date_start:
                print('Available Data ended {} *BEFORE* Selected Data start {}'
                      .format(last_date, date_start))
            print('&'*20, '\n')

        dfs.update({symbol: df})  # append symbol data to dictionary

    # sort caches by row_count in ascending order
    caches_sorted_by_row_count = sorted(caches, key=lambda tup: tup[1])

    # create df_caches dataframe
    df_caches = pd.DataFrame(
        columns=['symbol', 'row_count', 'first_date', 'last_date',
                 'all_zeros_row_count'])

    #  write to df_caches
    for index, (symbol, row_count, first_date, last_date, all_zeros_row_count)\
            in enumerate(caches_sorted_by_row_count):
        df_caches.loc[index] = \
            symbol, row_count, first_date, last_date, all_zeros_row_count

    if verbose:
        # print('There are {} symbols in {}\n{}\n'
        #       .format(len(symbol_names), file_symbol_names, symbol_names))

        print('caches: ', '\n', caches, '\n')
        print('caches sorted by row_count:', '\n',
              caches_sorted_by_row_count, '\n')

        # print symbol dataframes stored in dictionary dfs
        print('='*30, ' dfs dataframes ', '='*30, '\n')
        for symbol in symbols:
            print('symbol: ', symbol)
            print(dfs[symbol].head(2), '\n')
            print(dfs[symbol].tail(2), '\n')
        print('='*76, '\n'*2)
        print('df_caches: ')
        print(df_caches.head(2), '\n')
        print(df_caches.tail(2), '\n')
        print('{} symbols: {}\n'.format(len(symbols), symbols))

    print('- def get_pooled_symbol_data(file_symbol_names, \
path_pooled_symbol_data, file_ext=".csv", date_start=None, date_end=None, \
verbose=False)')
    print('{}\n'.format('-'*78))

    return dfs, df_caches, symbols


def plot_symbols(symbols, path_data_dump, date_start_limit=None,
                 date_end_limit=None, iloc_offset=None):
    """Plots symbol in symbols list. Program ps1n2 created dictionary
    dfs_symbols_OHLCV_download which has symbols' OHLCV data.
    The dictionary is stored in directory path_data_dump. This program plots
    the symbols using data from the dictionary.

    (Copied from def dates_within_limits)
    Order of hierarchy to determine Start and End dates:
    1. given date if it is not None
    2. iloc_offset if it is not None

    start = date_start_limit
    end = date_end_limit
    date_start_limit sets the upper (i.e. oldest) limit.
    date_end_limit sets the lower (i.e. newest) limit.
    date_start is the oldest index date that is within the limits.
    date_end is the newest index date that is within the limits.

    START   END    ILOC-OFFSET  Return->  START DATE             END DATE
    ---------------------------------------------------------------------------
    start   end	   iloc_offset	          date_start	         date_end
    start   end	   none                   date_start	         date_end
    start   none   iloc_offset	          date_start	         date_start+iloc_offset  # NOQA
    start   none   none                   date_start	         index[-1]
    none	end	   iloc_offset	          date_end-iloc_offset   date_end
    none	end	   none                   index[0]	             date_end
    none	none   iloc_offset	          index[0]	             index[-1]
    none	none   none                   index[0]	             index[-1]

    Args:
        symbols(list[str]): list of symbols, e.g. ['AAPL', 'GOOGL', ...]
        path_symbol_data(str): directory path of the symbol data
        date_start_limit(str): limit on the start date in format 'yyyy-mm-dd'
        date_end_limit(str): limit on the end date in format 'yyyy-mm-dd'
        iloc_offset(int): index offset
    
    Return:
        cache_returned_by_plot:
            symbol(str): stock symbol, e.g. 'AMZN'
            date(str): date, e.g. '2018-01-05'
            UI_MW_short[-1](float): last value of UI_MW_short,
                Ulcer-Index / Moving-Window-Short
            UI_MW_long[-1](float): last value of UI_MW_long
                Ulcer-Index / Moving-Window-Long
            diff_UI_MW_short[-1](float): last value of diff_UI_MW_short
                difference of last two 'Ulcer-Index / Moving-Window-Short'
            diff_UI_MW_long[-1](float): last value of diff_UI_MW_long
                difference of last two 'Ulcer-Index / Moving-Window-Long'
    """

    from myUtils import pickle_load, dates_within_limits
    from CStick_DD_OBV_UIMW_Diff_UIMW_cache import candlestick

    print('+'*15 + '  ps4_plot_symbols(symbols, path_data_dump)  ' + '+'*15
          + '\n')

    # directories and file names
    dfs_filename = 'dfs_symbols_OHLCV_download'
    index_symbol = 'XOM'
    # list of cache. Stores cache returned by candlestick plot
    caches_returned_by_plot = []
    # read dfs
    dfs = pickle_load(path_data_dump, dfs_filename)
    date_index_all_dates = dfs[index_symbol].index
    # get iloc of date_end_limit in XOM's date index
    date_start, date_end, iloc_date_start, iloc_date_end = \
        dates_within_limits(date_index_all_dates, date_start_limit,
                            date_end_limit, iloc_offset)

    for symbol in symbols:
        df = dfs[symbol]
        df = df[date_start:date_end]
        print('plot symbol {} from {} to {}\n'
              .format(symbol, date_start, date_end))
        # cache_returned_by_plot = candlestick(symbol, df, plot_chart=False)
        cache_returned_by_plot = candlestick(symbol, df, plot_chart=True)
        caches_returned_by_plot.append(cache_returned_by_plot)

    print('{}\n'.format('-'*78))

    return caches_returned_by_plot


def print_symbol_data(symbols, path_symbols_data, date_end_limit=None,
                      rows_OHLCV_display=5, tag=''):
    """Prints symbol data stored in MktCap2b.csv for stocks, and AUMtop1200.csv
    for ETFs. It also prints symbols' OHLCV data.

    Args:
        symbols([str]): list of symbols, e.g. ['AAPL', 'GOOGL', ...]
        path_symbols_data(str): directory path of the symbol data
        date_end_limit(str): date of last row of OHLCV data, default: None will
          use today's date, e.g. 'yyyy-mm-dd'
        rows_OHLCV_display(int): number of rows of OHLCV data to display
        tag(str): information tag, use as needed

    Return:
        symbols_stock(str): list of stock symbols
        symbols_etf(str): list of etf symbols

    v2 add tag
    v1 return symbols_stock, symbols_etf
    """

    import pandas as pd
    import datetime as dt  # NOQA
    from myUtils import pickle_load

    pd.set_option('display.width', 300)
    # pd.set_option('display.max_columns', 7)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_colwidth", 30)

    print('+'*28 + '  print_symbol_data  ' + '+'*28 + '\n')
    if date_end_limit is None:
        date_end_limit = dt.date.today().strftime('%Y-%m-%d')

    path_data_dump = path_symbols_data + 'VSCode_dump/'

    # symbol data files downloaded from Vanguard stock screen or ETF.com
    path_source = \
        'G:/My Drive/stocks/MktCap2b_AUMtop1200/source/'
        # 'C:/Users/ping/Google Drive/stocks/MktCap2b_AUMtop1200/source/'

    # file_etfs = 'AUMtop1200.csv'  # largest 1200 ETF by AUM from etf.com
    # file_stocks = 'MktCap2b.csv'  # stock with market-cap over 2b from Vanguard
    file_etfs = '2021_AUMtop1200.csv'  # largest 1200 ETF by AUM from etf.com
    file_stocks = '2021_MktCapTop1200.csv'  # stock with market-cap over 2b from TDAmeritrade

    # etf_cols = ['Symbol', 'Fund Name', 'Segments', 'AUM', 'Spread %']
    etf_cols = ['Symbol', 'Fund Name', 'Segments', 'AUM', 'Spread %']
    # load ETF data
    df_etf = \
        pd.read_csv(path_source + file_etfs, encoding='unicode_escape')
    df_etf = df_etf[etf_cols]  # subset of selected columns
    all_etf_symbols = df_etf['Symbol'].tolist()  # list of ETF symbols
    dict_etf = {'etf': all_etf_symbols}
    symbols_etf = []  # symbols that are etfs

    # stock_cols = ['Symbol', 'Company name', 'Market cap(mil)',
    #               'Sector', 'Industry']
    stock_cols = ['Symbol', 'Company Name', 'Market Cap ($M)',
                  'Sector', 'Industry']
    # load stock data
    df_stock = \
        pd.read_csv(path_source + file_stocks, encoding='unicode_escape')
    df_stock = df_stock[stock_cols]  # subset of selected columns
    all_stock_symbols = df_stock['Symbol'].tolist()  # list of stock symbols
    dict_stock = {'stock': all_stock_symbols}
    symbols_stock = []  # symbols that are stocks

    symbols_OHLCV = 'dfs_symbols_OHLCV_download'  # OHLCV data for all symbols
    dfs_OHLCV = pickle_load(path_data_dump, symbols_OHLCV)  # load OHLCV data

    for symbol in symbols:  # print symbol data
        df = dfs_OHLCV[symbol]  # OHLCV for symbols
        # subset OHLCV to date_end_limit
        df = df.loc[:date_end_limit].sort_values(by='date', ascending=False)
        if symbol in dict_etf.get('etf'):
            symbols_etf.append(symbol)
            print('Symbol {} is an ETF {}'.format(symbol, tag))
            print(df_etf.loc[df_etf['Symbol'] == symbol])
        elif symbol in dict_stock.get('stock'):
            symbols_stock.append(symbol)
            print('Symbol {} is a stock {}'.format(symbol, tag))
            print(df_stock.loc[df_stock['Symbol'] == symbol])

        print('\ndfs_OHLCV[{}].tail(rows_OHLCV_display):\n{}'
              .format(symbol, df.head(rows_OHLCV_display)))
        print('*'*78, '\n')  # next symbol

    print('{}\n'.format('-'*78))
    return symbols_stock, symbols_etf


def pd_lists_intersection(list_of_lists, a_list, verbose:False):
    """List_of_list is a column or a row of a pandas dataframe where the items
    are lists. The function returns a lists-of-list that is the intersection
    the list in each dataframe item and the a_list.

    Args:
        list_of_lists([str]): [['AAPL', 'GOOGL'], ..., ['MRNA', 'GLD', 'IAU']]
        a_list([str]): list of symbols, e.g. ['AAPL', 'GOOGL', ...]

    Return:
        a lists-of-list that is the intersection the list in each dataframe
        item and the a_list where each list is sorted.
    """

    if verbose:
        print('+'*26 + '  pd_lists_intersection  ' + '+'*26 + '\n')
    # result_list_of_lists = []
    # for list_item in list_of_lists:
    #     intersect = list(set(list_item).intersection(a_list))
    #     intersect.sort()
    #     result_list_of_lists.append(intersect)
    # return result_list_of_lists

    if verbose:
        print('{}\n'.format('-'*78))
    return [
        sorted(list(set(list_item).intersection(a_list)))
        for list_item in list_of_lists
    ]


def get_symbol_OHLCV(dir_path, file_symbols, date_start=None,
                     date_end=None, pct_limit_all_zeros_rows=1/252*100,
                     verbose=False):
    # region docstring Deprecated, use get_symbol_data, it's faster
    # v1 return of df_symbols_close
    """The function reads symbols in file_symbols and writes the corresponding
    OHLCV data into dataframes. It writes symbols' closing prices to dataframe
    df_symbols_close. If the symbol has all-zeros rows and percentage of the
    all-zeros rows are below pct_limit_all_zeros_rows, the function will
    replace the all-zeros rows with data from adjacent rows. If the all-zeros
    rows are over pct_limit_all_zeros_rows, the function will not replace them
    with adjacent rows. The dataframes are subsetted to date_start:date_end,
    and write to dictionary dfs.

    Args:
        dir_path(str): directory path of the symbol data
        file_symbols(str'): text file name for the symobls,
            e.g. 'MktCap2b_AUMtop1200.txt'
        date_start(str): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        date_end(str): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        pct_limit_all_zeros_rows(float): percent limit of all-zeros rows in
            symbol's dataframe.
        verbose(True or False): prints output if True, default False

    Return:
        dfs(dict): dictionary of dataframes, e.g. dfs['QQQ']
        df_row_count(dataframe): dataframe with columns: symbol, row_count,
            first_date, last_date, all_zeros_row_count; sorted by row_counts
        symbols(str): list of symbols read from text file,
            e.g. ['AMZN', 'QQQ', 'SPY', ...]
        symbols_over_limit(list): list of symbols that have their
            all-zeros rows over pct_limit_all_zeros_rows
        symbols_last_row_zeros(list): list of symbols with all-zeros last row
        symbols_last_two_rows_zeros(list): list of symbols with all-zeros last
            two rows
        df_symbols_close(dataframe): dataframe with date index and symbols'
            closing prices as columns, and symbols as column names
    """
    # endregion docstring

    import pandas as pd

    print('+'*29 + '  get_symbol_OHLCV  ' + '+'*29 + '\n')
    # region get list of symbols
    with open(file_symbols, 'r') as f:  # get symbols from text file
        symbols = [line.strip() for line in f]  # remove whitespace char
    if verbose:
        print('symbols: {}\n'.format(symbols))
        print("len(symbols) before removing '' from symbol text file: {}"
              .format(len(symbols)))
    # removes '' in list of symbols, a blank line in text file makes '' in list
    symbols = list(filter(None, symbols))
    if verbose:
        print("len(symbols) after removing '' from symbol text file: {}"
              .format(len(symbols)))
    # endregion get list of symbols

    # column names of .csv files
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume']
    dfs = {}  # dictionary of dataframe for symbol data: date, open,...
    # list-of-tuples:
    #  symbol, row count, first_date, last_date, all_zeros_row_count
    symbols_all_zeros_rows = []
    symbols_n_values_over_limit = {}  # key=symbol, value=pct_all_zero_row
    symbols_last_row_zeros = []  # symbols with all zeros in last row
    symbols_last_two_rows_zeros = []  # sym. with all zeros in last two rows

    # create dataframe to store symbols' closing prices
    df_symbols_close = pd.DataFrame()

    for symbol in symbols:
        # use 1st column "date" as index
        df = pd.read_csv(dir_path + symbol + '.csv', names=col_names,
                         parse_dates=True, index_col=0)

        # get the symbol's date index and close column
        df_temp = df.close  # pandas series
        df_temp.rename(symbol, inplace=True)  # rename series name to symbol

        if symbol == symbols[0]:
            # first symbol, write close column to the empty dataframe
            df_symbols_close = pd.concat([df_temp], axis=1)
        else:
            # not the first symbol, append close column to dataframe
            df_symbols_close = pd.concat([df_symbols_close, df_temp], axis=1)

        # region over_limit
        len_df = len(df)
        date_index = df.index.strftime('%Y-%m-%d')
        # df where rows are all zeros
        df_rows_of_zeros = df[(df.T == 0).all()]
        # row count of all-zeros rows
        len_df_rows_of_zeros = len(df_rows_of_zeros)
        pct_all_zero_row = 100 * len_df_rows_of_zeros / len_df
        over_limit = \
            True if pct_all_zero_row > pct_limit_all_zeros_rows else False

        if verbose:
            print('\n{} has {} all-zeros rows out of {} total rows or {:5.3f}%'  # NOQA
                  .format(symbol, len_df_rows_of_zeros, len_df,
                          pct_all_zero_row))

        if len_df_rows_of_zeros > 0:
            # convert DatetimeIndex to list ['2019-01-03', ..., '2020-01-03']
            dates_all_zero_row = \
                list(df_rows_of_zeros.index.strftime('%Y-%m-%d'))
            iloc_all_zero_row = []  # iloc for dates in dates_all_zero_row
            for date in dates_all_zero_row:
                iloc_date = date_index.get_loc(date)
                iloc_all_zero_row.append(iloc_date)
            # check zeros in last row or zeros in last-two rows
            date_last = date_index[-1]
            iloc_last = date_index.get_loc(date_last)
            if iloc_last in iloc_all_zero_row:
                symbols_last_row_zeros.append(symbol)
            if iloc_last - 1 in iloc_all_zero_row:
                symbols_last_two_rows_zeros.append(symbol)

        if over_limit:  # all-zero-rows are over limit
            # update dict
            symbols_n_values_over_limit[symbol] = pct_all_zero_row
            if verbose:
                print('{} has pct_all_zero_row = {:5.3f}%, and is OVER pct_limit_all_zeros_rows {:5.3f}'  # NOQA
                      .format(symbol, pct_all_zero_row,
                              pct_limit_all_zeros_rows))
                print('All-zero-row will NOT BE REPLACED with adjacent row')
        # not over_limit, replace all-zero-rows with adjacent row
        elif len_df_rows_of_zeros > 0:
            if verbose:
                print('{} has pct_all_zero_row = {:5.3f}%, and is UNDER pct_limit_all_zeros_rows {:5.3f}%'  # NOQA
                    .format(symbol, pct_all_zero_row,
                            pct_limit_all_zeros_rows))
                print('All-zero-row will be REPLACED with adjacent row')
            for i in iloc_all_zero_row:
                if i == 0:  # 1st row is zeros, replace with 2nd row
                    if verbose:
                        print('\nReplace all-zero-first-row with next row')  # NOQA
                        print('&&&&&&& Before replacing all-zero-first-row &&&&&&')  # NOQA
                        print(df.iloc[i:i+2])
                    # 1st row is zeros, replace with 2nd row
                    df.iloc[i] = df.iloc[i+1]
                    if verbose:
                        print('\n&&&&&&& After replacing all-zero-first-row &&&&&&&')  # NOQA
                        print(df.iloc[i:i+2])
                else:  # 1st row is not zeros, replace with previous row
                    if verbose:
                        print('\nReplace all-zero-row with previous row')  # NOQA                    
                        print('&&&&&&& Before replacing all zero row &&&&&&&')
                        print(df.iloc[i-1:i+1])
                    # not the 1st row, replace with previous row
                    df.iloc[i] = df.iloc[i-1]
                    if verbose:
                        print('\n&&&&&&& After replacing all zero row &&&&&&&')
                        print(df.iloc[i-1:i+1])
            if verbose:
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        # get dictionary's keys as a list of symbols
        symbols_over_limit = [*symbols_n_values_over_limit]
        if verbose:
            print('='*25, '\n')
            print('*+'*20)
            print('pct_limit_all_zeros_rows: {:5.3f}%'
                  .format(pct_limit_all_zeros_rows))
            print('symbols_n_values_over_limit: {}'
                  .format(symbols_n_values_over_limit))
            print('symbols_over_limit: {}'.format(symbols_over_limit))
            print('symbols_last_row_zeros: {}'.format(symbols_last_row_zeros))
            print('symbols_last_two_rows_zeros: {}'
                  .format(symbols_last_two_rows_zeros))
        # endregion over_limit

        if verbose:
            print('symbol: {}'.format(symbol))
            print('df.head(3):\n{}\n'.format(df.head(3)))
            print('df.tail(3):\n{}\n'.format(df.tail(3)))

        row_count = df.shape[0]
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        # collect symbol's all zeros rows into a df
        df_rows_of_zeros = df[(df.T == 0).all()]
        # count of rows with all zeros
        all_zeros_row_count = len(df_rows_of_zeros)

        # i.e. all_zeros_row_count = ('XLK', 268, '2017-06-01', '2018-06-22')
        symbol_all_zeros_rows = (symbol, row_count, first_date, last_date,
                                 all_zeros_row_count)  # tuple
        symbols_all_zeros_rows.append(symbol_all_zeros_rows)  # list-of-tuples

        df = df[date_start:date_end]  # select date range dates

        if df.empty:
            print('&'*20)
            print('df {} is empty:\n{}'.format(symbol, df))
            print('Available Data: df[{}:{}]:'
                  .format(first_date, last_date))
            print('Selected Data:  df[{}:{}]:'
                  .format(date_start, date_end))
            if last_date < date_start:
                print('Available Data ended {} *BEFORE* Selected Data start {}'
                      .format(last_date, date_start))
            print('&'*20, '\n')

        dfs.update({symbol: df})  # append symbol data to dictionary

    # sort symbols_all_zeros_rows by row_count in ascending order
    symbols_all_zeros_rows_sorted_by_row_count = \
        sorted(symbols_all_zeros_rows, key=lambda tup: tup[1])

    # create df_row_count dataframe
    df_row_count = pd.DataFrame(
        columns=['symbol', 'row_count', 'first_date', 'last_date',
                 'all_zeros_row_count'])

    #  write to df_row_count
    for index, (symbol, row_count, first_date, last_date, all_zeros_row_count)\
            in enumerate(symbols_all_zeros_rows_sorted_by_row_count):
        df_row_count.loc[index] = \
            symbol, row_count, first_date, last_date, all_zeros_row_count

    if verbose:
        print('\n')
        print('+'*9, ' START OUTPUT: def get_symbol_data_from_dir ', '+'*9)
        print('\n')
        print('Number of symbols' + ' files in directory = ',
              len(symbols), '\n')
        print('symbols_all_zeros_rows: ', '\n', symbols_all_zeros_rows, '\n')
        print('symbols_all_zeros_rows sorted by row_count:', '\n',
              symbols_all_zeros_rows_sorted_by_row_count, '\n')
        # print symbol dataframes stored in dictionary dfs
        print('='*30, ' dfs dataframes ', '='*30, '\n')
        for symbol in symbols:
            print('symbol: ', symbol)
            print(dfs[symbol].head(2), '\n')
            print(dfs[symbol].tail(2), '\n')
        print('='*76, '\n'*2)
        print('df_row_count: ')
        print(df_row_count.head(2), '\n')
        print(df_row_count.tail(2), '\n')
        print('symbols: ', symbols, '\n')
        print('+'*9, ' END OUTPUT: def get_symbol_data_from_dir ',
              '+'*10, '\n'*2)

    print('{}\n'.format('-'*78))
    return dfs, df_row_count, symbols, symbols_over_limit, \
        symbols_last_row_zeros, symbols_last_two_rows_zeros, df_symbols_close


def get_symbol_data(dir_path, file_symbols, date_start=None, date_end=None,
                    verbose=False):
    """The function reads symbols in file_symbols and writes the corresponding
    OHLCV data into dataframes subsetted to date_start:date_end, and writes
    the dataframes to dictionary dfs. Dates are inclusive. None returns all
    data.

    Args:
        dir_path(str): directory path of the symbol data
        file_symbols(str): path + file-name to symbols-text-file
        date_start(str): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        date_end(st): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        verbose(True or False): prints output if True, default False

    Return:
        dfs(dict}: dictionary of symbols' OHLCV dataframes, e.g. dfs['QQQ']
        df_symbols_row_count(dataframe): dataframe with columns: symbol,
            row_count, first_date, last_date, all_zeros_row_count;
            sorted by row_counts
        symbols[str]: list of symbols; ['AMZN', 'QQQ', 'SPY', ...]
    """

    import pandas as pd

    print('+'*29 + '  get_symbol_data  ' + '+'*29 + '\n')
    # region get list of symbols
    with open(file_symbols, 'r') as f:  # get symbols from text file
        # removes leading and trailing whitespaces from the string
        symbols = [line.strip() for line in f]

    if verbose:
        print('symbols: {}\n'.format(symbols))
        print("len(symbols) before removing '' from symbol text file: {}"
              .format(len(symbols)))

    # removes '' in list of symbols, a blank line in text file makes '' in list
    symbols = list(filter(None, symbols))
    if verbose:
        print("len(symbols) after removing '' from symbol text file: {}"
              .format(len(symbols)))
    # endregion get list of symbols

    # column names of .csv files
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume']
    dfs = {}  # dictionary of dataframe for symbol data: date, open,...
    caches = []  # list for row counts, first date and last date for symbol

    for symbol in symbols:
        # use 1st column "date" as index
        df = pd.read_csv(dir_path + symbol + '.csv', names=col_names,
                         parse_dates=True, index_col=0)
        row_count = df.shape[0]
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        # collect symbol's all zeros rows into a df
        df_rows_of_zeros = df[(df.T == 0).all()]
        # count of rows with all zeros
        all_zeros_row_count = len(df_rows_of_zeros)
        # type(cache) = <class 'tuple'>
        # cache = ('XLK', 268, Timestamp('2017-06-01 00:00:00'),
        #          Timestamp('2018-06-22 00:00:00'))
        cache = (symbol, row_count, first_date, last_date,
                 all_zeros_row_count)  # tuple
        caches.append(cache)  # list
        df = df[date_start:date_end]  # select date range dates

        if df.empty:
            print('&'*20)
            print('df {} is empty:\n{}'.format(symbol, df))
            print('Available Data: df[{}:{}]:'
                  .format(first_date, last_date))
            print('Selected Data:  df[{}:{}]:'
                  .format(date_start, date_end))
            if last_date < date_start:
                print('Available Data ended {} *BEFORE* Selected Data start {}'
                      .format(last_date, date_start))
            print('&'*20, '\n')

        dfs.update({symbol: df})  # append symbol data to dictionary

    # sort caches by row_count in ascending order
    caches_sorted_by_row_count = sorted(caches, key=lambda tup: tup[1])

    # create df_symbols_row_count dataframe
    df_symbols_row_count = pd.DataFrame(
        columns=['symbol', 'row_count', 'first_date', 'last_date',
                 'all_zeros_row_count'])

    #  write to df_symbols_row_count
    for index, (symbol, row_count, first_date, last_date, all_zeros_row_count)\
            in enumerate(caches_sorted_by_row_count):
        df_symbols_row_count.loc[index] = \
            symbol, row_count, first_date, last_date, all_zeros_row_count

    if verbose:
        print('\n')
        print('+'*9, ' START OUTPUT: def get_symbol_data ', '+'*9)
        print('\n')
        print('Number of symbols = {}\n'.format(len(symbols)))
        print('caches: ', '\n', caches, '\n')
        print('caches sorted by row_count:', '\n',
              caches_sorted_by_row_count, '\n')

        # print symbol dataframes stored in dictionary dfs
        print('='*30, ' dfs dataframes ', '='*30, '\n')
        for symbol in symbols:
            print('symbol: ', symbol)
            print(dfs[symbol].head(2), '\n')
            print(dfs[symbol].tail(2), '\n')
        print('='*76, '\n'*2)

        print('df_symbols_row_count: ')
        print(df_symbols_row_count.head(2), '\n')
        print(df_symbols_row_count.tail(2), '\n')

        print('symbols: ', symbols, '\n')
        print('+'*9, ' END OUTPUT: def get_symbol_data_from_dir ',
              '+'*10, '\n'*2)

    print('{}\n'.format('-'*78))
    return dfs, df_symbols_row_count, symbols


def get_symbol_data_2(dir_path, file_symbols, date_start=None, date_end=None,
                      index_symbol='XOM', verbose=False):
    """The function reads symbols in file_symbols and writes their OHLCV data
    into dataframes and store the dataframes to dictionary dfs. Dates are
    inclusive. None returns all data in csv file.

    The function create df_symbols_close, a dataframe with index_symbol's
    date as index, symbols' closing price in columns, and symbols as column
    names. Missing prices are filled with NaN.

    The function also create df_symbols_volume, a dataframe with index_symbol's
    date as index, symbols' daily volume in columns, and symbols as column
    names. Missing prices are filled with NaN.

    v1 add back index_symbol to symbols_with_csv_data

    Args:
        dir_path(str): directory path of the symbol data
        file_symbols(str): path + file-name to symbols-text-file
        date_start(str): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        date_end(st): 'yyyy-mm-dd', 'yyyy', 'yyyy-mm', None
        index_symbol(str): date index for df_symbols_close, 'XOM'
        verbose(True or False): prints output if True, default False

    Return:
        dfs(dict}: dictionary of symbols' OHLCV dataframes with
            all available dates, e.g. dfs['QQQ']
        symbols_with_csv_data[str]: list of symbols in file_symobls with csv
            data, e.g. ['AMZN', 'QQQ', 'SPY', ...]
        symbols_no_csv_data[str]: list of symbols in file_symobls without csv
            data, e.g. ['AMZN', 'QQQ', 'SPY', ...]
        df_symbols_close(df): Dataframe with index_symbol's date as index,
            symbols' closing price in columns, and symbols as column names.
            Missing prices are filled with NaN.
        df_symbols_volume(df): Dataframe with index_symbol's date as index,
            symbols' daily volume in columns, and symbols as column names.
            Missing volume are filled with NaN.
    """

    import pandas as pd

    print('+'*29 + '  get_symbol_data  ' + '+'*29 + '\n')
    # region get list of symbols
    with open(file_symbols, 'r') as f:  # get symbols from text file
        # removes leading and trailing whitespaces from the string
        symbols = [line.strip() for line in f]

    if verbose:
        print('symbols: {}\n'.format(symbols))
        print("len(symbols) before removing '' from symbol text file: {}"
              .format(len(symbols)))
    # removes '' in list of symbols, a blank line in text file makes '' in list
    symbols = list(filter(None, symbols))

    if verbose:
        print("len(symbols) after removing '' from symbol text file: {}"
              .format(len(symbols)))
    # endregion get list of symbols

    # column names of .csv files
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume']
    dfs = {}  # dictionary of dataframe for symbol data: date, open,...
    df_symbols_close = pd.DataFrame()  # df with symbols' close price
    # read index_symbol's OHLCV
    df = pd.read_csv(dir_path + index_symbol + '.csv', names=col_names,
                     parse_dates=True, index_col=0)
    # append symbol data to dict AS READ from csv file, no NaN fill
    dfs.update({index_symbol: df})

    # pandas series, select date range dates
    df_temp_close = df.close[date_start:date_end]
    # rename series name to symbol
    df_temp_close.rename(index_symbol, inplace=True)
    # write index_symbol's close column to the empty dataframe
    df_symbols_close = pd.concat([df_temp_close], axis=1)

    df_temp_volume = df.volume[date_start:date_end]
    # rename series name to symbol
    df_temp_volume.rename(index_symbol, inplace=True)
    # write index_symbol's close column to the empty dataframe
    df_symbols_volume = pd.concat([df_temp_volume], axis=1)

    symbols_dropped_index_symbol = symbols
    # index_symbol already loaded into dfs, don't need to it again
    symbols_dropped_index_symbol.remove(index_symbol)

    symbols_no_csv_data = []
    for symbol in symbols_dropped_index_symbol:
        try:
            df = pd.read_csv(dir_path + symbol + '.csv', names=col_names,
                             parse_dates=True, index_col=0)
            # append symbol data to dict AS READ from csv file, no NaN fill
            dfs.update({symbol: df})

            # create df with all symbols' closing prices,
            #  missing data is filled with NaN
            df_temp_close = df.close  # pandas series
            # rename series name to symbol
            df_temp_close.rename(symbol, inplace=True)
            # dates not in index_symbol's date index are filled with NaN
            df_symbols_close = \
                pd.merge(df_symbols_close, df_temp_close,
                         how='left', on='date')

            # create df with all symbols' volumes
            #  missing data is filled with NaN
            df_temp_volume = df.volume  # pandas series
            # rename series name to symbol
            df_temp_volume.rename(symbol, inplace=True)
            # dates not in index_symbol's date index are filled with NaN
            df_symbols_volume = \
                pd.merge(df_symbols_volume, df_temp_volume,
                         how='left', on='date')
        except Exception:
            symbols_no_csv_data.append(symbol)

    symbols_with_csv_data = list(set(symbols) - set(symbols_no_csv_data))
    symbols_with_csv_data.insert(0, index_symbol)  # add back index_symbol
    print('{}\n'.format('-'*78))

    return dfs, symbols_with_csv_data, symbols_no_csv_data, \
        df_symbols_close, df_symbols_volume


def append_df(row_data, columns_drop_duplicate,
              path_data_dump, filename_df,
              path_archive, filename_df_archive, verbose=False):
    """The function appends a row of data, in dictionary row_data, to dataframe
    filename_df if the file exists in directory path_data_dump.  It archives
    the df to directory path_archive to the name filename_df_archives.  If
    filename_df does not exist, it creates a dataframe with the row_data.

    Args:
        row_data(dict): dict with key as column name and value as cell value
        columns_drop_duplicate(list): list of column names to check for
          duplicate rows.
        path_data_dump(str): path to load and dump filename_df
        filename_df(str): name of df to append the data
        path_archive_(str): path to archive filename_df before append data
        filename_archive_df(str): name of df to be archived

    Return:
        df(dataframe): df with the appended data
    """

    import os
    import pandas as pd
    from myUtils import pickle_load, pickle_dump

    # if df does not exists, create empty df with columns
    if not(os.path.exists(path_data_dump + filename_df)):  # does df exists?
        # df does not exists
        df = pd.DataFrame([row_data])  # create df
        # save to path_data_dump
        pickle_dump(df, path_data_dump, filename_df)
        # save to archive, overwrite if same file name exists
        pickle_dump(df, path_archive, filename_df_archive)
    else:
        # df exists, archive 'df_ps1n2_results'
        df = pickle_load(path_data_dump, filename_df)  # load df
        # save to archive, overwrite if same file name exists
        pickle_dump(df, path_archive, filename_df_archive)
        # append results, ignore_index=True generates the next index value
        df = df.append([row_data], ignore_index=True)
        # data cleaning: drop duplicate rows
        #  drop_duplicates gets "TypeError: unhashable type: 'list'"
        #  if df has list, use astype(str) to convert df to string
        index_to_keep = df.astype(str)\
            .drop_duplicates(subset=columns_drop_duplicate, keep='last').index
        # use loc, loc is for the index, .iloc is for the position
        #  iloc results in IndexError: positional indexers are out-of-bounds
        df_dropped_duplicate = df.loc[index_to_keep]
        if verbose:
            print('df before drop duplicate(shape = {}):\n{}\n'
                  .format(df.shape, df))
            print('df index to keep:\n{}\n'.format(index_to_keep))
            print('df after drop duplicate(shape = {}):\n{}\n'
                  .format(df_dropped_duplicate.shape,
                          df_dropped_duplicate))
        df = df_dropped_duplicate
        # save df as pickle and csv files
        pickle_dump(df, path_data_dump, filename_df)
        df.to_csv(path_data_dump + filename_df + '.csv')

    return df


def append_df_2(row_data, columns_drop_duplicate,
                path_data_dump, filename_df,
                path_archive, filename_df_archive, verbose=False):
    """The function appends a row of data, in dictionary row_data, to dataframe
    filename_df if the file exists in directory path_data_dump.  It archives
    the df to directory path_archive to the name filename_df_archives.  If
    filename_df does not exist, it creates a dataframe with the row_data.

    Args:
        row_data(dict): dict with key as column name and value as cell value
        columns_drop_duplicate(list): list of column names to check for
          duplicate rows.
        path_data_dump(str): path to load and dump filename_df
        filename_df(str): name of df to append the data
        path_archive_(str): path to archive filename_df before append data
        filename_archive_df(str): name of df to be archived

    Return:
        df(dataframe): df with the appended data
    """
    # v2 replaced pickle_dump, pickle_load with df.to_pickle, pd.read_pickle


    import os
    import pandas as pd
    from myUtils import pickle_load, pickle_dump


    my_file = path_data_dump + filename_df
    my_archive = path_archive + filename_df_archive

    # if df does not exists, create empty df with columns
    if not(os.path.exists(path_data_dump + filename_df)):  # does df exists?
        # df does not exists
        df = pd.DataFrame([row_data])  # create df
        # save to path_data_dump

        # pickle_dump(df, path_data_dump, filename_df)
        df.to_pickle(my_file)
        df.to_csv(my_file + '.csv')
        # save to archive, overwrite if same file name exists
        # pickle_dump(df, path_archive, filename_df_archive)
        df.to_pickle(my_archive)

    else:
        # df exists, archive 'df_ps1n2_results'
        # df = pickle_load(path_data_dump, filename_df)  # load df
        df = pd.read_pickle(my_file)

        # save to archive, overwrite if same file name exists
        # pickle_dump(df, path_archive, filename_df_archive)
        df.to_pickle(my_archive)
        # append results, ignore_index=True generates the next index value
        df = df.append([row_data], ignore_index=True)
        # data cleaning: drop duplicate rows
        #  drop_duplicates gets "TypeError: unhashable type: 'list'"
        #  if df has list, use astype(str) to convert df to string
        index_to_keep = df.astype(str)\
            .drop_duplicates(subset=columns_drop_duplicate, keep='last').index
        # use loc, loc is for the index, .iloc is for the position
        #  iloc results in IndexError: positional indexers are out-of-bounds
        df_dropped_duplicate = df.loc[index_to_keep]
        if verbose:
            print('df before drop duplicate(shape = {}):\n{}\n'
                  .format(df.shape, df))
            print('df index to keep:\n{}\n'.format(index_to_keep))
            print('df after drop duplicate(shape = {}):\n{}\n'
                  .format(df_dropped_duplicate.shape,
                          df_dropped_duplicate))
        df = df_dropped_duplicate
        # save df as pickle and csv files
        # pickle_dump(df, path_data_dump, filename_df)
        df.to_pickle(my_file)
        df.to_csv(my_file + '.csv')

    return df


def symb_perf_stats_vectorized_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, Std_UI, CAGR, CAGR_Std, CAGR_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pandas.core.indexes.base.Index): stock symbols
        period_yr(float): years, (days in dataframe) / 252
        drawdown(dataframe): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pandas.series float64): ulcer-index
        max_drawdown(pandas series float64): maximum drawdown from peak
        returns_std(pandas series float64): standard deviation of daily returns
        Std_UI(pandas series float64): returns_std / UI
        CAGR(pandas series float64): compounded annual growth rate
        CAGR_Std(pandas series float64): CAGR / returns_std
        CAGR_UI(pandas series float64): CAGR / UI
    """

    import numpy as np

    symbols = df_symbols_close.columns
    df_symbols_returns = df_symbols_close / df_symbols_close.shift(1) - 1
    # standard deviation divisor is N - ddof
    returns_std = df_symbols_returns.std(ddof=1)
    # +++ SET RETURNS OF FIRST ROW = 0,
    #  otherwise drawdown calculation starts with the second row
    df_symbols_returns.iloc[0] = 0
    cum_returns = (1 + df_symbols_returns).cumprod()
    drawdown = cum_returns.div(cum_returns.cummax()) - 1
    max_drawdown = drawdown.min()
    UI = np.sqrt(np.sum(np.square(drawdown)) / len(drawdown))
    Std_UI = returns_std / UI
    period_yr = len(df_symbols_close) / 252  # 252 trading days per year
    CAGR = (df_symbols_close.iloc[-1] / df_symbols_close.iloc[0]) \
        ** (1 / period_yr) - 1
    CAGR_Std = CAGR / returns_std
    CAGR_UI = CAGR / UI

    return symbols, period_yr, drawdown, UI, max_drawdown, \
        returns_std, Std_UI, CAGR, CAGR_Std, CAGR_UI


def symb_perf_stats_vectorized_v1_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, Std_UI, CAGR, CAGR_Std, CAGR_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pandas.core.indexes.base.Index): stock symbols
        period_yr(float): years, (days in dataframe) / 252
        drawdown(numpy array): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pandas.series float64): ulcer-index
        max_drawdown(pandas series float64): maximum drawdown from peak
        returns_std(pandas series float64): standard deviation of daily returns
        Std_UI(pandas series float64): returns_std / UI
        CAGR(pandas series float64): compounded annual growth rate
        CAGR_Std(pandas series float64): CAGR / returns_std
        CAGR_UI(pandas series float64): CAGR / UI
    """
    # v1 convert drawdown from pandas series to numpy array

    import numpy as np

    symbols = df_symbols_close.columns
    df_symbols_returns = df_symbols_close / df_symbols_close.shift(1) - 1
    # standard deviation divisor is N - ddof
    returns_std = df_symbols_returns.std(ddof=1)
    # +++ SET RETURNS OF FIRST ROW = 0,
    #  otherwise drawdown calculation starts with the second row
    df_symbols_returns.iloc[0] = 0
    cum_returns = (1 + df_symbols_returns).cumprod()
    
    drawdown = cum_returns.div(cum_returns.cummax()) - 1
    # convert from pandas Series into a NumPy array
    drawdown = np.array(drawdown)

    max_drawdown = drawdown.min()
    UI = np.sqrt(np.sum(np.square(drawdown)) / len(drawdown))
    Std_UI = returns_std / UI
    period_yr = len(df_symbols_close) / 252  # 252 trading days per year
    CAGR = (df_symbols_close.iloc[-1] / df_symbols_close.iloc[0]) \
        ** (1 / period_yr) - 1
    CAGR_Std = CAGR / returns_std
    CAGR_UI = CAGR / UI

    return symbols, period_yr, drawdown, UI, max_drawdown, \
        returns_std, Std_UI, CAGR, CAGR_Std, CAGR_UI


def symb_perf_stats_vectorized_v2_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pandas.core.indexes.base.Index): stock symbols
        period_yr(float): years, (days in dataframe) / 252
        drawdown(pandas dataframe): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pandas.series float64): ulcer-index
        max_drawdown(pandas series float64): maximum drawdown from peak
        returns_std(pandas series float64): standard deviation of daily returns
        returns_std_div_UI(pandas series float64): returns_std / UI
        CAGR(pandas series float64): compounded annual growth rate
        CAGR_div_Std(pandas series float64): CAGR / returns_std
        CAGR_div_UI(pandas series float64): CAGR / UI
    """
    # https://stackoverflow.com/questions/44933518/how-to-remove-runtimewarning-errors-from-code
    # https://stackoverflow.com/questions/10519237/python-how-to-avoid-numpy-runtimewarning-in-function-definition/10519607#10519607
    #  suppress RuntimeWarning divide by zero    
    # v1 convert drawdown from pandas series to numpy array
    # v2 do calculation in numpy array

    import numpy as np
    import pandas as pd

    dates = df_symbols_close.index
    symbols = df_symbols_close.columns
    len_df = len(df_symbols_close)

    arr = df_symbols_close.to_numpy()
    arr_returns = arr / np.roll(arr, 1, axis=0) - 1
    arr_returns_std = np.std(
        arr_returns[1:, :], axis=0, ddof=1
    )  # drop first row
    arr_returns[0] = 0  # set first row to 0
    arr_cum_returns = (1 + arr_returns).cumprod(
        axis=0
    )  # cumulative product of column elements
    # accumulative max value of column elements
    arr_drawdown = (
        arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
    )
    arr_max_drawdown = arr_drawdown.min(axis=0)
    arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
    arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
    returns_std_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
    period_yr = len_df / 252  # 252 trading days per year
    arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1
    arr_CAGR_div_arr_Std = arr_CAGR / arr_returns_std
    arr_CAGR_div_arr_UI = arr_CAGR / arr_UI

    # convert numpy array to dataframe, add date index and symbols as column names
    drawdown = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)
    # convert numpy array to pandas series, add symbols as index
    UI = pd.Series(arr_UI, index=symbols)
    max_drawdown = pd.Series(arr_max_drawdown, index=symbols)
    returns_std = pd.Series(arr_returns_std, index=symbols)
    CAGR = pd.Series(arr_CAGR, index=symbols)
    CAGR_div_Std = pd.Series(arr_CAGR_div_arr_Std, index=symbols)
    CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        drawdown,
        UI,
        max_drawdown,
        returns_std,
        returns_std_div_UI,
        CAGR,
        CAGR_div_Std,
        CAGR_div_UI,
    )


def symb_perf_stats_vectorized_v3_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pandas.core.indexes.base.Index): stock symbols
        period_yr(float): years, (days in dataframe) / 252
        drawdown(pandas dataframe): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pandas.series float64): ulcer-index
        max_drawdown(pandas series float64): maximum drawdown from peak
        returns_std(pandas series float64): standard deviation of daily returns
        returns_std_div_UI(pandas series float64): returns_std / UI
        CAGR(pandas series float64): compounded annual growth rate
        CAGR_div_Std(pandas series float64): CAGR / returns_std
        CAGR_div_UI(pandas series float64): CAGR / UI
    """
    # https://stackoverflow.com/questions/44933518/how-to-remove-runtimewarning-errors-from-code
    # https://stackoverflow.com/questions/10519237/python-how-to-avoid-numpy-runtimewarning-in-function-definition/10519607#10519607
    #  suppress RuntimeWarning divide by zero    
    # v1 convert drawdown from pandas series to numpy array
    # v2 do calculation in numpy array
    # v3 ignore divide by zero error

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero
    with np.errstate(divide='ignore'):

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1
        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0, ddof=1
        )  # drop first row
        arr_returns[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns).cumprod(
            axis=0
        )  # cumulative product of column elements
        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )
        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        returns_std_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
        period_yr = len_df / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1
        arr_CAGR_div_arr_Std = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI

        # convert numpy array to dataframe, add date index and symbols as column names
        drawdown = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)
        # convert numpy array to pandas series, add symbols as index
        UI = pd.Series(arr_UI, index=symbols)
        max_drawdown = pd.Series(arr_max_drawdown, index=symbols)
        returns_std = pd.Series(arr_returns_std, index=symbols)
        CAGR = pd.Series(arr_CAGR, index=symbols)
        CAGR_div_Std = pd.Series(arr_CAGR_div_arr_Std, index=symbols)
    CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        drawdown,
        UI,
        max_drawdown,
        returns_std,
        returns_std_div_UI,
        CAGR,
        CAGR_div_Std,
        CAGR_div_UI,
    )


def symb_perf_stats_vectorized_v4_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pandas.core.indexes.base.Index): stock symbols
        period_yr(float): years, (days in dataframe) / 252
        drawdown(pandas dataframe): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pandas.series float64): ulcer-index
        max_drawdown(pandas series float64): maximum drawdown from peak
        returns_std(pandas series float64): standard deviation of daily returns
        returns_std_div_UI(pandas series float64): returns_std / UI
        CAGR(pandas series float64): compounded annual growth rate
        CAGR_div_Std(pandas series float64): CAGR / returns_std
        CAGR_div_UI(pandas series float64): CAGR / UI
    """
    # https://stackoverflow.com/questions/44933518/how-to-remove-runtimewarning-errors-from-code
    # https://stackoverflow.com/questions/10519237/python-how-to-avoid-numpy-runtimewarning-in-function-definition/10519607#10519607
    #  suppress RuntimeWarning divide by zero    
    # v1 convert drawdown from pandas series to numpy array
    # v2 do calculation in numpy array
    # v3 ignore divide by zero error
    # v4 corrected period_yr calculation, was len_df / 252, should be period_yr = (len_df - 1) / 252 
    #    corrected CAGR calculation, was arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1, should be
    #    arr_CAGR = (arr[-1] / arr[0]) ** (period_yr) - 1
  

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero
    with np.errstate(divide='ignore'):

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1
        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0, ddof=1
        )  # drop first row
        arr_returns[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns).cumprod(
            axis=0
        )  # cumulative product of column elements
        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )
        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        returns_std_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
        period_yr = (len_df - 1) / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (period_yr) - 1
        arr_CAGR_div_arr_Std = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI

        # convert numpy array to dataframe, add date index and symbols as column names
        drawdown = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)
        # convert numpy array to pandas series, add symbols as index
        UI = pd.Series(arr_UI, index=symbols)
        max_drawdown = pd.Series(arr_max_drawdown, index=symbols)
        returns_std = pd.Series(arr_returns_std, index=symbols)
        CAGR = pd.Series(arr_CAGR, index=symbols)
        CAGR_div_Std = pd.Series(arr_CAGR_div_arr_Std, index=symbols)
        CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        drawdown,
        UI,
        max_drawdown,
        returns_std,
        returns_std_div_UI,
        CAGR,
        CAGR_div_Std,
        CAGR_div_UI,
    )


def symb_perf_stats_vectorized_v5_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pandas.core.indexes.base.Index): stock symbols
        period_yr(float): years, (days in dataframe) / 252
        drawdown(pandas dataframe): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pandas.series float64): ulcer-index
        max_drawdown(pandas series float64): maximum drawdown from peak
        returns_std(pandas series float64): standard deviation of daily returns
        returns_std_div_UI(pandas series float64): returns_std / UI
        CAGR(pandas series float64): compounded annual growth rate
        CAGR_div_Std(pandas series float64): CAGR / returns_std
        CAGR_div_UI(pandas series float64): CAGR / UI
    """
    # https://stackoverflow.com/questions/44933518/how-to-remove-runtimewarning-errors-from-code
    # https://stackoverflow.com/questions/10519237/python-how-to-avoid-numpy-runtimewarning-in-function-definition/10519607#10519607
    #  suppress RuntimeWarning divide by zero    
    # v1 convert drawdown from pandas series to numpy array
    # v2 do calculation in numpy array
    # v3 ignore divide by zero error
    # v4 corrected period_yr calculation, was len_df / 252, should be period_yr = (len_df - 1) / 252 
    #    corrected CAGR calculation, was arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1, should be
    #    arr_CAGR = (arr[-1] / arr[0]) ** (period_yr) - 1
    # v5     

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero
    with np.errstate(divide='ignore'):

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1


        arr_returns_mean = np.mean(
            arr_returns[1:, :], axis=0
        )  # drop first row
        # # # convert numpy array to dataframe, add date index and symbols as column names
        # # returns = pd.DataFrame(arr_returns, index=dates, columns=symbols)





        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0, ddof=1
        )  # drop first row


        # arr_returns[0] = 0  # set first row to 0
        # arr_cum_returns = (1 + arr_returns).cumprod(
        #     axis=0
        # )  # cumulative product of column elements
        arr_returns_0 = arr_returns
        arr_returns_0[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns_0).cumprod(
            axis=0
        )  # cumulative product of column elements




        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )
        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        returns_std_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
        period_yr = (len_df - 1) / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (period_yr) - 1
        arr_CAGR_div_arr_Std = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI


        grp_sum_CAGR = arr_CAGR.sum()  # sum of group's CAGR to be compared with SPY CAGR


        # convert numpy array to dataframe, add date index and symbols as column names
        returns = pd.DataFrame(arr_returns, index=dates, columns=symbols)


        # convert numpy array to dataframe, add date index and symbols as column names
        drawdown = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)
        # convert numpy array to pandas series, add symbols as index
        UI = pd.Series(arr_UI, index=symbols)
        max_drawdown = pd.Series(arr_max_drawdown, index=symbols)


        returns_mean = pd.Series(arr_returns_mean, index=symbols)


        returns_std = pd.Series(arr_returns_std, index=symbols)
        CAGR = pd.Series(arr_CAGR, index=symbols)
        CAGR_div_Std = pd.Series(arr_CAGR_div_arr_Std, index=symbols)
        CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        returns,
        drawdown,
        UI,
        max_drawdown,
        returns_mean,
        returns_std,
        returns_std_div_UI,
        CAGR,
        CAGR_div_Std,
        CAGR_div_UI,
        grp_sum_CAGR,
    )


def symb_perf_stats_vectorized_v6_(df_symbols_close):

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero
    with np.errstate(divide='ignore'):

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1


        arr_returns_mean = np.mean(
            arr_returns[1:, :], axis=0
        )  # drop first row


        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0, ddof=1
        )  # drop first row


        arr_returns_0 = arr_returns
        arr_returns_0[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns_0).cumprod(
            axis=0
        )  # cumulative product of column elements
        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )
        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        retnStd_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
        period_yr = (len_df - 1) / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1
        arr_CAGR_div_arr_Std = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI


        grp_SumCAGR = arr_CAGR.sum()  # sum of group's CAGR to be compared with SPY CAGR


        # convert numpy array to dataframe, add date index and symbols as column names
        retn = pd.DataFrame(arr_returns, index=dates, columns=symbols)


        # convert numpy array to dataframe, add date index and symbols as column names
        DD = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)  # drawdown
        # convert numpy array to pandas series, add symbols as index
        UI = pd.Series(arr_UI, index=symbols)
        MDD = pd.Series(arr_max_drawdown, index=symbols)  # max drawdown
        # retnMean = arr_returns_mean
        retnMean = pd.Series(arr_returns_mean, index=symbols)       
        retnStd = pd.Series(arr_returns_std, index=symbols)
        CAGR = pd.Series(arr_CAGR, index=symbols)
        CAGR_div_retnStd = pd.Series(arr_CAGR_div_arr_Std, index=symbols)
        CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        retn,
        DD,
        UI,
        MDD,
        retnMean,
        retnStd,
        retnStd_div_UI,
        CAGR,
        CAGR_div_retnStd,
        CAGR_div_UI,
        grp_SumCAGR,
    )


def symb_perf_stats_vectorized_v7_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pd.Index, i.e. list): stock symbols
        period_yr(float): (len(df) - 1) / 252
        retn(pd.df, float): daily returns
            with date index and symbols as column names        
        DD(pd.df, float): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pd.Series, float): ulcer-index
        MDD(pd.Series, float): maximum drawdown from peak
        retnMean(pd.series, float): mean return for each symbol
        retnStd(pd.Series, float): standard deviation of daily returns
        retnStd_div_UI(pd.Series, float): retnStd / UI
        CAGR(pd.Series, float): compounded annual growth rate
        CAGR_div_retnStd(pd.Series, float): CAGR / retnsStd
        CAGR_div_UI(pd.Series, float): CAGR / UI
        grp_MeanCAGR(float): CAGR mean for the symbols
        grp_StdCAGR(float): CAGR standard deviation for the symbols
    """

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero
    with np.errstate(divide='ignore'):

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1
        arr_returns_mean = np.mean(
            arr_returns[1:, :], axis=0
        )  # drop first row
        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0
        )  # drop first row
        arr_returns_0 = arr_returns
        arr_returns_0[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns_0).cumprod(
            axis=0
        )  # cumulative product of column elements
        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )
        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        retnStd_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
        period_yr = (len_df - 1) / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1
        arr_CAGR_div_arr_Std = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI
        grp_MeanCAGR = arr_CAGR.mean()  # mean of group's CAGR to be compared with SPY CAGR
        # grp_StdCAGR = arr_CAGR.std()  # std of group's CAGR to be compared with SPY CAGR
        grp_StdCAGR = np.std(arr_CAGR)  # std of group's CAGR to be compared with SPY CAGR
        # convert numpy array to dataframe, add date index and symbols as column names
        retn = pd.DataFrame(arr_returns, index=dates, columns=symbols)
        # convert numpy array to dataframe, add date index and symbols as column names
        DD = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)  # drawdown
        # convert numpy array to pandas series, add symbols as index
        UI = pd.Series(arr_UI, index=symbols)
        MDD = pd.Series(arr_max_drawdown, index=symbols)  # max drawdown
        # retnMean = arr_returns_mean
        retnMean = pd.Series(arr_returns_mean, index=symbols)       
        retnStd = pd.Series(arr_returns_std, index=symbols)
        CAGR = pd.Series(arr_CAGR, index=symbols)
        CAGR_div_retnStd = pd.Series(arr_CAGR_div_arr_Std, index=symbols)
        CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        retn,
        DD,
        UI,
        MDD,
        retnMean,
        retnStd,
        retnStd_div_UI,
        CAGR,
        CAGR_div_retnStd,
        CAGR_div_UI,
        grp_MeanCAGR,
        grp_StdCAGR,
    )


def symb_perf_stats_vectorized_v8(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pd.Index, i.e. list): stock symbols
        period_yr(float): (len(df) - 1) / 252
        retn(pd.df, float): daily returns
            with date index and symbols as column names        
        DD(pd.df, float): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        UI(pd.Series, float): ulcer-index
        MDD(pd.Series, float): maximum drawdown from peak
        retnMean(pd.series, float): mean return for each symbol
        retnStd(pd.Series, float): standard deviation of daily returns
        retnStd_div_UI(pd.Series, float): retnStd / UI
        CAGR(pd.Series, float): compounded annual growth rate
        CAGR_div_retnStd(pd.Series, float): CAGR / retnsStd
        CAGR_div_UI(pd.Series, float): CAGR / UI
        grp_retnStd_div_UI(np.array):
          [arr_returns_std_div_arr_UI.mean,
           np.std(arr_returns_std_div_arr_UI)
           arr_returns_std_div_arr_UI.mean / np.std(arr_returns_std_div_arr_UI)]        
        grp_CAGR(np.array): [arr_CAGR.mean, np.std(arr_CAGR), arr_CAGR.mean / np.std(arr_CAGR)]
        grp_CAGR_div_retnStd(np.array):
          [arr_CAGR_div_arr_retnStd.mean,
           np.std(arr_CAGR_div_arr_retnStd),
           arr_CAGR_div_arr_retnStd.mean / np.std(arr_CAGR_div_arr_retnStd)]
        grp_CAGR_div_UI(np.array):
          [arr_CAGR_div_arr_UI.mean,
           np.std(arr_CAGR_div_arr_UI),
           arr_CAGR_div_arr_UI.mean / np.std(arr_CAGR_div_arr_UI)]     
    """
    # v8 place 0.000001 in the last row where columns in the 2d drawdown array are all zeros
    #    added grp_retnStd_div_UI, grp_CAGR, grp_CAGR_div_retnStd, grp_CAGR_div_UI

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero and 
    # with np.errstate(divide='ignore')
    # ignore divide by zero encountered in scalar divide
    # It occurs when df_eval returns 1 symbol and grp(CAGR) is 0.0
    # with np.errstate(divide='ignore', invalid='ignore')
    with np.errstate(divide='ignore', invalid='ignore'):  # changeed 2023-03-17  

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1
        arr_returns_mean = np.mean(
            arr_returns[1:, :], axis=0
        )  # drop first row
        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0
        )  # drop first row
        arr_returns_0 = arr_returns
        arr_returns_0[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns_0).cumprod(
            axis=0
        )  # cumulative product of column elements
        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )

        # place 0.000001 in the last row of columns with all zeros in drawdown array
        #   prevent UI calculation to be 0, and CAGR/UI to be infinite 
        _cols_no_drawdown = np.all(arr_drawdown==0, axis=0)  # True if column is all 0, i.e. the symbol does not have drawdown
        _arr_drawdown_last_row = arr_drawdown[-1]  # get the last row
        for _i, _cond in enumerate(_cols_no_drawdown):
            if _cond:
                _arr_drawdown_last_row[_i] = 0.000001  # replace zero with a small value  
        arr_drawdown[-1] = _arr_drawdown_last_row  # replace the last row

        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        retnStd_div_UI = pd.Series(arr_returns_std_div_arr_UI, index=symbols)
        period_yr = (len_df - 1) / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1
        arr_CAGR_div_arr_retnStd = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI

        # store group's (retnStd/UI) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_returns_std_div_arr_UI.mean()
        _grp_std = np.std(arr_returns_std_div_arr_UI)
        grp_retnStd_div_UI = [_grp_mean, _grp_std, _grp_mean/_grp_std]
        # store group's (CAGR) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_CAGR.mean()
        _grp_std = np.std(arr_CAGR)
        grp_CAGR = [_grp_mean, _grp_std, _grp_mean/_grp_std]
        # store group's (CAGR/Std) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_CAGR_div_arr_retnStd.mean()
        _grp_std = np.std(arr_CAGR_div_arr_retnStd)
        grp_CAGR_div_retnStd = [_grp_mean, _grp_std, _grp_mean/_grp_std]
        # store group's (CAGR/UI) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_CAGR_div_arr_UI.mean()
        _grp_std = np.std(arr_CAGR_div_arr_UI)
        grp_CAGR_div_UI = [_grp_mean, _grp_std, _grp_mean/_grp_std]

        # convert numpy array to dataframe, add date index and symbols as column names
        retn = pd.DataFrame(arr_returns, index=dates, columns=symbols)
        # convert numpy array to dataframe, add date index and symbols as column names
        DD = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)  # drawdown
        # convert numpy array to pandas series, add symbols as index
        UI = pd.Series(arr_UI, index=symbols)
        MDD = pd.Series(arr_max_drawdown, index=symbols)  # max drawdown
        # retnMean = arr_returns_mean
        retnMean = pd.Series(arr_returns_mean, index=symbols)       
        retnStd = pd.Series(arr_returns_std, index=symbols)
        CAGR = pd.Series(arr_CAGR, index=symbols)
        CAGR_div_retnStd = pd.Series(arr_CAGR_div_arr_retnStd, index=symbols)
        CAGR_div_UI = pd.Series(arr_CAGR_div_arr_UI, index=symbols)

    return (
        symbols,
        period_yr,
        retn,
        DD,
        UI,
        MDD,
        retnMean,
        retnStd,
        retnStd_div_UI,
        CAGR,
        CAGR_div_retnStd,
        CAGR_div_UI,
        grp_retnStd_div_UI,     
        grp_CAGR,
        grp_CAGR_div_retnStd,
        grp_CAGR_div_UI,
    )



def symb_perf_stats_vectorized_v9_(df_symbols_close):
    """Takes dataframe of symbols' close and returns symbols, period_yr,
       drawdown, UI, max_drawdown, returns_std, returns_std_div_UI, CAGR, CAGR_div_Std, CAGR_div_UI
       https://stackoverflow.com/questions/36750571/calculate-max-draw-down-with-a-vectorized-solution-in-python
       http://www.tangotools.com/ui/ui.htm
       Calculation CHECKED against: http://www.tangotools.com/ui/UlcerIndex.xls
       Calculation VERIFIED in: symb_perf_stats_vectorized.ipynb

    Args:
        df_symbols_close(dataframe): dataframe with date as index,
          symbol's close in columns, and symbols as column names.

    Return:
        symbols(pd.Index, i.e. list): stock symbols
        period_yr(float): (len(df) - 1) / 252
        df_retn(pd.df, float): daily returns
            with date index and symbols as column names        
        df_drawdown(pd.df, float): drawdown from peak, 0.05 means 5% drawdown,
            with date index and symbols as column names
        arr_UI(np.array): ulcer-index for each symbol
        arr_max_drawdown(np.array): maximum drawdown from peak for each symbol
        arr_returns_mean(np.array): mean return for each symbol
        arr_returns_std(np.array): standard deviation of daily returns for each symbol
        arr_returns_std_div_arr_UI(np.array): arr_returns_std / arr_UI for each symbol
        arr_CAGR(np.array): compounded annual growth rate for each symbol
        arr_CAGR_div_arr_retnStd(np.array): arr_CAGR / arr_returns_std for each symbol
        arr_CAGR_div_arr_UI(np.array): arr_CAGR / arr_UI for each symbol
        grp_CAGR(np.array): [arr_CAGR.mean, np.std(arr_CAGR), arr_CAGR.mean / np.std(arr_CAGR)]
        grp_CAGR_div_retnStd(np.array):
          [arr_CAGR_div_arr_retnStd.mean,
           np.std(arr_CAGR_div_arr_retnStd),
           arr_CAGR_div_arr_retnStd.mean / np.std(arr_CAGR_div_arr_retnStd)]
        grp_CAGR_div_UI(np.array):
          [arr_CAGR_div_arr_UI.mean,
           np.std(arr_CAGR_div_arr_UI),
           arr_CAGR_div_arr_UI.mean / np.std(arr_CAGR_div_arr_UI)]        
    """
    # v8 place 0.000001 in the last row where columns in the 2d drawdown array are all zeros
    # v9 added grp_CAGR, grp_CAGR_div_Std, grp_CAGR_div_UI, change return items from pd.Series to nparrays 

    import numpy as np
    import pandas as pd

    # ignore divide by zero in case arr_drawdown is zero
    with np.errstate(divide='ignore'):

        dates = df_symbols_close.index
        symbols = df_symbols_close.columns
        len_df = len(df_symbols_close)

        arr = df_symbols_close.to_numpy()
        arr_returns = arr / np.roll(arr, 1, axis=0) - 1
        arr_returns_mean = np.mean(
            arr_returns[1:, :], axis=0
        )  # drop first row
        arr_returns_std = np.std(
            arr_returns[1:, :], axis=0
        )  # drop first row
        arr_returns_0 = arr_returns
        arr_returns_0[0] = 0  # set first row to 0
        arr_cum_returns = (1 + arr_returns_0).cumprod(
            axis=0
        )  # cumulative product of column elements
        # accumulative max value of column elements
        arr_drawdown = (
            arr_cum_returns / np.maximum.accumulate(arr_cum_returns, axis=0) - 1
        )

        # place 0.000001 in the last row of columns with all zeros in drawdown array
        #   prevent UI calculation to be 0, and CAGR/UI to be infinite 
        _cols_no_drawdown = np.all(arr_drawdown==0, axis=0)  # True if column is all 0, i.e. the symbol does not have drawdown
        _arr_drawdown_last_row = arr_drawdown[-1]  # get the last row
        for _i, _cond in enumerate(_cols_no_drawdown):
            if _cond:
                _arr_drawdown_last_row[_i] = 0.000001  # replace zero with a small value  
        arr_drawdown[-1] = _arr_drawdown_last_row  # replace the last row

        arr_max_drawdown = arr_drawdown.min(axis=0)
        arr_UI = np.sqrt(np.sum(np.square(arr_drawdown), axis=0) / len_df)
        arr_returns_std_div_arr_UI = arr_returns_std / arr_UI
        period_yr = (len_df - 1) / 252  # 252 trading days per year
        arr_CAGR = (arr[-1] / arr[0]) ** (1 / period_yr) - 1
        arr_CAGR_div_arr_retnStd = arr_CAGR / arr_returns_std
        arr_CAGR_div_arr_UI = arr_CAGR / arr_UI


        # store group's (retnStd/UI) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_returns_std_div_arr_UI.mean()
        _grp_std = np.std(arr_returns_std_div_arr_UI)
        grp_returns_std_div_UI = [_grp_mean, _grp_std, _grp_mean/_grp_std]

        # store group's (CAGR) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_CAGR.mean()
        _grp_std = np.std(arr_CAGR)
        grp_CAGR = [_grp_mean, _grp_std, _grp_mean/_grp_std]
        # store group's (CAGR/Std) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_CAGR_div_arr_retnStd.mean()
        _grp_std = np.std(arr_CAGR_div_arr_retnStd)
        grp_CAGR_div_retnStd = [_grp_mean, _grp_std, _grp_mean/_grp_std]
        # store group's (CAGR/UI) mean, std, mean/std to be compared with SPY
        _grp_mean = arr_CAGR_div_arr_UI.mean()
        _grp_std = np.std(arr_CAGR_div_arr_UI)
        grp_CAGR_div_UI = [_grp_mean, _grp_std, _grp_mean/_grp_std]

        # convert numpy array to dataframe, add date index and symbols as column names
        df_retn = pd.DataFrame(arr_returns, index=dates, columns=symbols)  # daily return
        # convert numpy array to dataframe, add date index and symbols as column names
        df_drawdown = pd.DataFrame(arr_drawdown, index=dates, columns=symbols)  # daily drawdown

    return (
        symbols,
        period_yr,
        df_retn,
        df_drawdown,
        arr_UI,
        arr_max_drawdown,
        arr_returns_mean,    
        arr_returns_std,
        arr_returns_std_div_arr_UI,
        arr_CAGR,
        arr_CAGR_div_arr_retnStd,
        arr_CAGR_div_arr_UI,
        grp_returns_std_div_UI,    
        grp_CAGR,
        grp_CAGR_div_retnStd,
        grp_CAGR_div_UI,
    )



def concat_df(df, columns_drop_duplicate, file_df, file_df_archive, verbose=False):
    """The function concatenate dateframes df and file_df, pickles df,
    archives file_df, and returns concatenated dataframe df_concat.

    Args:
        df(dataframe): dataframe to be concatendated with columns same as
         file_df
        columns_drop_duplicate(list): list of column names to check for
          duplicate rows.
        file_df(str): path and name of df to be concatendated
        file_df_archive(str): path and name of df to be archived

    Return:
        df_concat(dataframe): concatendated dataframe
    """

    import os
    import pandas as pd

    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_colwidth", 20)

    if not (os.path.exists(file_df)):  # does df exists?
        # df does not exists, save, archive and to_csv df
        if verbose:
            print(f"{file_df} does not exists")
        df.to_pickle(file_df)
        df.to_pickle(file_df_archive)
        df.to_csv(file_df + ".csv")
        df_concat = df
    else:
        # df exists, read and archive, overwrite if same file name exists
        if verbose:
            print(f"{file_df} exists")
        df_current = pd.read_pickle(file_df)
        df_current.to_pickle(file_df_archive)
        # df_concat = pd.concat([df, df_current], ignore_index=True)
        df_concat = pd.concat([df_current, df], ignore_index=True)
        if verbose:
            print(f"len(df_concat) before drop_duplicates: {len(df_concat)}")
            print(f"df_concat:\n{df_concat}\n")
        df_concat = df_concat.drop_duplicates(
            subset=columns_drop_duplicate, keep="last",
        )
        if verbose:
            print(f"len(df_concat) after drop_duplicates: {len(df_concat)}")
            print(f"df_concat:\n{df_concat}\n")

        df_concat  = df_concat.sort_values(
            by=['date_end', 'symbol'], ascending=False
        )
        df_concat.to_pickle(file_df)
        df_concat.to_csv(file_df + ".csv")

    return df_concat


def concat_df_2(df, my_column, my_value, file_df, file_df_archive, verbose=False):  # NOQA
    """The function loads file in file_df and drops rows with values equal to
    my value in my_column. Then, it concatenate dateframes df and file_df,
    pickles df, archives file_df, and returns concatenated dataframe df_concat.

    Args:
        df(dataframe): dataframe to be concatendated with columns same as
         file_df
        my_column(str): column name to check for rows to drop.
        my_value(str): drop rows in my_column when value is equal to my_value
        file_df(str): path and name of df to be concatendated
        file_df_archive(str): path and name of df to be archived

    Return:
        df_concat(dataframe): concatendated dataframe
    """

    import os
    import pandas as pd

    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_colwidth", 20)

    if not (os.path.exists(file_df)):  # does df exists?
        # df does not exists, save, archive and to_csv df
        if verbose:
            print(f"{file_df} does not exists")
        df.to_pickle(file_df)
        df.to_pickle(file_df_archive)
        df.to_csv(file_df + ".csv")
        df_concat = df
    else:
        # df exists, read and archive, overwrite if same file name exists
        if verbose:
            print(f"{file_df} exists")
        df_current = pd.read_pickle(file_df)
        df_current.to_pickle(file_df_archive)
        print(
            f"df_current before drop [{my_column}] == "
            f"{my_value}:\n{df_current}\n"
        )
        index_drop = df_current[df_current[my_column] == my_value].index
        print(f"index_drop: {index_drop}\n")
        df_current.drop(index_drop, inplace=True)
        print(
            f"df_current after drop [{my_column}] == "
            f"{my_value}:\n{df_current}\n"
        )
        df_concat = pd.concat([df_current, df], ignore_index=True)
        print(f"df_concat:\n{df_concat}\n")
        df_concat = df_concat.sort_values(
            by=["date_end", "symbol"], ascending=False
        )
        df_concat.to_pickle(file_df)
        df_concat.to_csv(file_df + ".csv")

    return df_concat


def NYSE_dates(date_pivot, len_list):
    """The function returns a list of NYSE trading dates. If len_list is
    positive, the list starts with a date that is equal or newer than
    date_pivot and subsequent NYSE trading dates. If len_list is negative,
    the list ends with a date that is equal or older than date_pivot and
    previous NYSE trading dates. The length of the list is equal to len_list.

    Args:
        date_pivot(str): date in format "2020-12-31"
        len_list(int): length of the return list, cannot equal zero.

    Return:
        dates_nyse(list): list of NYSE dates,
          e.g. ["2020-12-01", "2020-12-02", "2020-12-03"]
    """

    import datetime
    import datetime as dt
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    # buffer to convert calendar dates to trading dates
    delta_buffer = len_list * 3

    if len_list == 0:
        raise Exception(
            f"ERROR: len_list = {len_list}, "
            f"It must be a positive or negative interger."
        )
    elif len_list > 0:
        # returns NYSE dates that are equal or newer than date_pivot,
        # len(dates_nyse) = len_list
        start_date = date_pivot
        # convert to datetime
        dt_start = dt.datetime.strptime(start_date, "%Y-%m-%d")
        dt_end = dt_start + datetime.timedelta(days=delta_buffer)
        dates_nyse = nyse.valid_days(
            start_date=dt_start, end_date=dt_end
        ).strftime("%Y-%m-%d")
        dates_nyse = dates_nyse.tolist()[0:len_list]
    else:
        # returns NYSE dates that are equal or older than date_pivot,
        # len(dates_nyse) = len_list
        end_date = date_pivot
        # convert to datetime
        dt_end = dt.datetime.strptime(end_date, "%Y-%m-%d")
        dt_start = dt_end + datetime.timedelta(days=delta_buffer)
        dates_nyse = nyse.valid_days(
            start_date=dt_start, end_date=dt_end
        ).strftime("%Y-%m-%d")
        dates_nyse = dates_nyse.tolist()[len_list::]

    return dates_nyse

######################################
def yf_download_AdjOHLCV(symbols, verbose=False):
    """Download daily adjusted OHLCV data for symbols, and return dataframe df.
    To fetch ajusted OHLCV data for symbol 'SPY', use df['SPY'].

    Args:
        symbols(list): list of symbols(i.e. ['SPY',...,'AAPL'])
        verbose(bool): default False

    Return:
        df(dataframe): dataframe with adjusted OHLCV data for symbols,
                       To fetch OHLCV data for symbol 'SPY', use df['SPY'].
    """
    # https://stackoverflow.com/questions/69192215/python-yfinance-api-how-to-get-close-share-price-instead-of-adjusted-close-share

    import yfinance as yf

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def yf_download_AdjOHLCV(symbols, verbose=False)\n')

    df = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        # tickers = "SPY AAPL MSFT",
        tickers=symbols,
        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period="max",
        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval="1d",
        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by="ticker",
        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust=True,
        # download pre/post regular market hours data
        # (optional, default is False)
        prepost=False,
        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads=True,
        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy=None,
    )
    if verbose:    
        print('- def yf_download_AdjOHLCV(symbols, verbose=False)')    
        print('{}\n'.format('-'*78))

    return df

def yf_download_AdjOHLCV_noAutoAdj(symbols, verbose=False):
    """Download daily adjusted OHLCV data for symbols, and return dataframe df.
    To fetch ajusted OHLCV data for symbol 'SPY', use df['SPY'].

    Args:
        symbols(list): list of symbols(i.e. ['SPY',...,'AAPL'])
        verbose(bool): default False

    Return:
        df(dataframe): dataframe with adjusted OHLCV data for symbols,
                       To fetch OHLCV data for symbol 'SPY', use df['SPY'].
    """
    # https://stackoverflow.com/questions/69192215/python-yfinance-api-how-to-get-close-share-price-instead-of-adjusted-close-share

    import yfinance as yf

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def yf_download_AdjOHLCV_noAutoAdj(symbols, verbose=False)\n')

    df = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        # tickers = "SPY AAPL MSFT",
        tickers=symbols,
        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period="max",
        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval="1d",
        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by="ticker",
        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust=False,
        # download pre/post regular market hours data
        # (optional, default is False)
        prepost=False,
        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads=True,
        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy=None,
    )
    if verbose:    
        print('- def yf_download_AdjOHLCV(symbols, verbose=False)')    
        print('{}\n'.format('-'*78))

    return df

def yf_symbols_close_old(
    path_dir,
    path_data_dump,
    filename_pickled_df_OHLCV,
    verbose=False,
):
    """Reads adjusted OHLCV data for symbols in pickled df_OHLCV and gathers
       symbols' Close into df_symbols_close. It has a date index and the
       column names are the symbols. Rows and columns with all NaN are dropped.

       Usage: df_symbols_close['SPY'] returns a series with date index and
       SPY Close

    Args:
        path_dir(str): directory path of df_OHLCV
        path_data_dump(str): directory of pickled files
        filename_pickled_df_OHLCV(str): filename of df_OHLCV
        verbose(bool): prints output if True, default False

    Return:
        df_symbols_close(dataframe): dataframe with a date index and
            columns of symbols' Close, column names are symbols
        dates_dropped(DatetimeIndex): dates where all symbols' Close are NaN
        symbols_OHLCV(list): symbols in df_OHLCV
        symbols_dropped(list): symbols where their Close is all NaN
    """

    import pandas as pd
    from myUtils import pickle_load, pickle_dump

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def yf_symbols_close(path_dir, path_data_dump, filename_pickled_df_OHLCV)\n')

    df = pickle_load(
        path_data_dump, filename_pickled_df_OHLCV, verbose=verbose
    )
    # get list of symbols in df_OHLCV, list(df) is a list of tuples
    # e.g.: [('AAPL', 'Open')..('AAPL', 'Volume'),...
    #        ('ZZZZ', 'Open')..('ZZZZ', 'Volume')]
    symbols_OHLCV = list(set([i[0] for i in list(df)]))

    # write symbols' Close to df_symbols_close
    for sym in symbols_OHLCV:
        if (
            sym == symbols_OHLCV[0]
        ):  # create dataframe using the 1st symbol's Close
            df_symbols_close = pd.DataFrame(df[symbols_OHLCV[0]].close)
        else:  # concatenate Close on subsequent symbols
            df_symbols_close = pd.concat(
                [df_symbols_close, df[sym].close], axis=1
            )
    # rename column names from Close to symbol names
    df_symbols_close.set_axis(symbols_OHLCV, axis=1, inplace=True)

    if verbose:
        # drop rows and columns with all NaN
        print(f"df_symbols_close.info() before dropna:")
        print(df_symbols_close.info(), '\n')
    df_symbols_close_index_before_dropna = df_symbols_close.index
    # drop all NaN rows
    df_symbols_close = df_symbols_close.dropna(
        how="all", axis="index"
    )
    # drop all NaN columns
    df_symbols_close = df_symbols_close.dropna(
        how="all", axis="columns"
    )
    if verbose:
        print(f"df_symbols_close.info() after dropna:")
        print(df_symbols_close.info(), '\n')    
    # dates with all NaN in row
    dates_dropped = df_symbols_close_index_before_dropna.difference(
        df_symbols_close.index
    )
    # symbols (i.e. column names) with all NaN in column
    symbols_dropped = list(set(symbols_OHLCV) - set(list(df_symbols_close)))

    pickle_dump(
        df_symbols_close, path_data_dump, "df_symbols_close", verbose=verbose
    )
    pickle_dump(
        dates_dropped,
        path_data_dump,
        "df_OHLCV_dates_dropped",
        verbose=verbose,
    )
    pickle_dump(
        symbols_dropped,
        path_data_dump,
        "df_OHLCV_symbols_dropped",
        verbose=verbose,
    )
    if verbose:   
        print('- def yf_symbols_close(path_dir, path_data_dump, filename_pickled_df_OHLCV)')
        print('{}\n'.format('-'*78))

    return df_symbols_close, dates_dropped, symbols_OHLCV, symbols_dropped

def yf_symbols_close(
    path_dir,
    path_data_dump,
    filename_pickled_df_OHLCV,
    verbose=False,
):
    """Reads adjusted OHLCV data for symbols in pickled df_OHLCV and gathers
       symbols' Close into df_symbols_close. It has a date index and the
       column names are the symbols. Rows and columns with all NaN are dropped.

       Usage: df_symbols_close['SPY'] returns a series with date index and
       SPY Close

    Args:
        path_dir(str): directory path of df_OHLCV
        path_data_dump(str): directory of pickled files
        filename_pickled_df_OHLCV(str): filename of df_OHLCV
        verbose(bool): prints output if True, default False

    Return:
        df_symbols_close(dataframe): dataframe with a date index and
            columns of symbols' Close, column names are symbols
        dates_dropped(DatetimeIndex): dates where all symbols' Close are NaN
        symbols_OHLCV(list): symbols in df_OHLCV
        symbols_dropped(list): symbols where their Close is all NaN
    """

    import pandas as pd
    from myUtils import pickle_load, pickle_dump

    if verbose:
        print('\n{}'.format('='*78))
        print('+ def yf_symbols_close(path_dir, path_data_dump, filename_pickled_df_OHLCV)\n')

    df = pickle_load(
        path_data_dump, filename_pickled_df_OHLCV, verbose=verbose
    )
    # get list of symbols in df_OHLCV, list(df) is a list of tuples
    # e.g.: [('AAPL', 'Open')..('AAPL', 'Volume'),...
    #        ('ZZZZ', 'Open')..('ZZZZ', 'Volume')]
    symbols_OHLCV = list(set([i[0] for i in list(df)]))



    # # write symbols' Close to df_symbols_close
    # for sym in symbols_OHLCV:
    #     if (
    #         sym == symbols_OHLCV[0]
    #     ):  # create dataframe using the 1st symbol's Close
    #         df_symbols_close = pd.DataFrame(df[symbols_OHLCV[0]].close)
    #     else:  # concatenate Close on subsequent symbols
    #         df_symbols_close = pd.concat(
    #             [df_symbols_close, df[sym].close], axis=1
    #         )

    # # rename column names from Close to symbol names
    # df_symbols_close.set_axis(symbols_OHLCV, axis=1, inplace=True)

    df_symbols_close = df.xs('Close', level=1, axis=1)



    if verbose:
        # drop rows and columns with all NaN
        print(f"df_symbols_close.info() before dropna:")
        print(df_symbols_close.info(), '\n')
    df_symbols_close_index_before_dropna = df_symbols_close.index
    # drop all NaN rows
    df_symbols_close = df_symbols_close.dropna(
        how="all", axis="index"
    )
    # drop all NaN columns
    df_symbols_close = df_symbols_close.dropna(
        how="all", axis="columns"
    )
    if verbose:
        print(f"df_symbols_close.info() after dropna:")
        print(df_symbols_close.info(), '\n')
    # dates with all NaN in row
    dates_dropped = df_symbols_close_index_before_dropna.difference(
        df_symbols_close.index
    )
    # symbols (i.e. column names) with all NaN in column
    symbols_dropped = list(set(symbols_OHLCV) - set(list(df_symbols_close)))

    pickle_dump(
        df_symbols_close, path_data_dump, "df_symbols_close", verbose=verbose
    )
    pickle_dump(
        dates_dropped,
        path_data_dump,
        "df_OHLCV_dates_dropped",
        verbose=verbose,
    )
    pickle_dump(
        symbols_dropped,
        path_data_dump,
        "df_OHLCV_symbols_dropped",
        verbose=verbose,
    )
    if verbose:   
        print('- def yf_symbols_close(path_dir, path_data_dump, filename_pickled_df_OHLCV)')
        print('{}\n'.format('-'*78))

    return df_symbols_close, dates_dropped, symbols_OHLCV, symbols_dropped

def read_symbols_file(filename_symbols, verbose=False):
    """Read symbols in text file filename_symbols, clean it and return a list
    of symbols. The cleaning process consists of:
        Removing leading and trailing spaces
        Removing banks (i.e. '')
        Converting to all uppercase
        Removing duplicates
        Sort

    Args:
        filename_symbols(str): full path to a text file with symbol on each line
        verbose(bool): default False
        
    Return:
        symbols(list): list of symbols
    """

    if verbose:
        print('\n{}'.format('='*78))
        print('+ read_symbols_file(filename_symbols)\n')

    with open(filename_symbols, "r") as f:  # get symbols from text file
        # remove leading and trailing whitespaces
        symbols = [line.strip() for line in f]

    # removes '' in list of symbols, a blank line in text file makes '' in list
    symbols = list(filter(None, symbols))
    symbols = [symbol.upper() for symbol in symbols]  # convert to upper case
    symbols = list(set(symbols))  # remove duplicate symbols
    list.sort(symbols)

    if verbose:
        print('- read_symbols_file(filename_symbols)')
        print('{}'.format('-'*78))
    
    return symbols

def drop_symbols_all_NaN(df, verbose=False):
    """Reads symbols in multi-index df returned by yfinance with OHLCV.
       Drops columns in df with symbols that have all NaN values for OHLCV.

    Args:
        df(dataframe): yfinance multi-index dataframe with symbols OHLCV
        verbose(bool): default False
        
    Return:
        df(dataframe): Multi-index dataframe with symbols OHLCV.
            Symbols with all NaN OHLCV are dropped.
        symbols_OHLCV(list): symbols in df_OHLCV, sorted
        symbols_dropped(list): symbols dropped from df where their Close is all NaN
    """

    if verbose:
        print('\n{}'.format('='*78))
        print('+ drop_symbols_all_NaN(df)\n')

    # get column names(i.e. symbols) from multi-index
    #  column names are converted to all cap by yfinance download
    col_syms = list(set(df.columns.get_level_values(0)))
    col_syms.sort()
    if verbose:
        print(f"Symbols in dataframe: {col_syms}")
    syms_with_NaN = []
    for symbol in col_syms:
        # Check for symbols with all NaN dataframe
        # 1st .all returns a series of True/False, 2nd .all reduces to single True/False
        if df[symbol].isna().all().all():
            print(f"Dataframe for {symbol} is all NaN")
            syms_with_NaN.append(symbol)
    syms_with_NaN.sort()
    # print(f"Symbols with all NaN OHLCV: {syms_with_NaN}")
    print(f"Symbol's OHLCV are all NaN: {syms_with_NaN}")
    symbols_all = list(set([i[0] for i in list(df)]))
    symbols_all.sort()
    if verbose:
        print(f"df columns before dropping symbols with all NaN:\n{symbols_all}")
    for symbol in syms_with_NaN:
        # drop symbol from dataframe
        df.drop(columns=[symbol], axis=1, level=0, inplace=True)
    symbols_OHLCV = list(set([i[0] for i in list(df)]))
    symbols_OHLCV.sort()
    if verbose:    
        print(f"df columns after dropping symbols with all NaN:\n{symbols_OHLCV}\n")
    symbols_dropped = list(set(symbols_all) - set(symbols_OHLCV))

    if verbose:
        print('- drop_symbols_all_NaN(df)')
        print('{}'.format('-'*78))

    return df, symbols_OHLCV, symbols_dropped



def yf_candlestick_(symbol, df, plot_chart=True):
    """Plot candlestick chart and returns:
        symbol, date, UI_MW_short[-1], UI_MW_long[-1],
        diff_UI_MW_short[-1], diff_UI_MW_long[-1]

    Args:
        symbol(string): symbol for the dataframe df
        df(dataframe): dataframe with date index
                       and columns open, high, low, close, volume
        plot_chart(bool): plots chart if True, default is True

    Return:
        cache(tuple) of:
            symbol(str): stock symbol, e.g. 'AMZN'
            date(str): plot's last date, e.g. '2018-01-05'
            UI_MW_short[-1](float): last value of UI_MW_short
            UI_MW_long[-1](float): last value of UI_MW_long
            diff_UI_MW_short[-1](float): last value of diff_UI_MW_short
            diff_UI_MW_long[-1](float): last value of diff_UI_MW_long
    """
    # 2022-08-21 change OBV's EMA period from 10 to 50, i.e.  OBV_period = 50


    # Reference:
    # https://www.kaggle.com/arrowsx/crypto-currency-a-plotly-sklearn-tutorial
    # https://plotly.com/python/candlestick-charts/
    # http://mrjbq7.github.io/ta-lib/
    # https://www.somesolvedproblems.com/2018/10/how-to-customize-plotlys-modebar.html
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Candlestick.html#plotly.graph_objects.Candlestick
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.html
    # https://community.plotly.com/t/random-vertical-lines-extending-to-x-axis-on-line-chart/23462
    # https://plotly.com/python/hover-text-and-formatting/

    from plotly.offline import plot
    from myUtils import symb_perf_stats, OBV_calc, UI_MW
    import plotly.graph_objs as go
    import talib as ta
    import numpy as np

    # **** IMPORTANT: COMMENT-OUT LINES INCLUDING THE BRACKETS {...} ****
    # if no lines are between the brackets, chart will be plotted compressed
    # to the right side and leaving empty spaces on the left side
    # ********

    chart_path = "C:/Users/ping/Desktop/plotly/"  # directory to store charts
    chart_name = symbol + ".html"

    close = df.close.values  # convert dataframe to numpy array

    # +++++++++ panel layout ++++++++++
    x_annotation = 0.06  # text x position
    y_annotation_gap = 0.022  # text y spacing

    # panels' y positions
    y80_top = 0.99  # spike, height 1
    y80_btm = 0.98
    y70_top = 0.98  # price, height 38
    y70_txt = 0.96  # price, text
    y70_btm = 0.60
    # gap of 1
    y60_top = 0.59  # volume, height 5
    # y60_txt = .57  # volume, text
    y60_btm = 0.54
    # gap of 1
    y50_top = 0.53  # drawdown, height 5
    y50_txt = 0.52  # drawdown, text
    y50_btm = 0.48
    # gap of 1
    y40_top = 0.47  # OBV, height 5
    y40_txt = 0.46  # OBV, text
    y40_btm = 0.42
    # gap of 1
    y30_top = 0.41  # top, height 13
    y30_txt = 0.40  # top, text
    y30_btm = 0.28
    # gap of 1
    y20_top = 0.27  # middle, height 13
    y20_txt = 0.26  # middle, text
    y20_btm = 0.14
    # gap of 1
    y10_top = 0.13  # bottom, height 13
    y10_txt = 0.12  # bottom, text
    y10_btm = 0.0

    # +++++++++ chart colors ++++++++++
    # http://colormind.io/template/paper-dashboard/#
    INCREASING_COLOR = "#17BECF"
    DECREASING_COLOR = "#7F7F7F"
    BBANDS_COLOR = "#BDBCBC"
    LIGHT_COLOR = "#0194A2"
    DARK_COLOR = "#015C65"
    DRAWDOWN_COLOR = "#5B5170"

    # ++++++++++ moving window sizes +++++++++++
    window_short = 15
    window_long = 30

    # ++++++++++ indicators +++++++++++
    # ++ spike line ++
    # With Toggle Spike Lines is on, gives vertical lines down the chart
    #   when cursor is at top of chart
    spike_line = np.zeros(len(df.index))  # create a trace with zeros
    spike_line_color = DRAWDOWN_COLOR

    # ++ moving averages ++
    EMA_fast_period = 50  # moving average rolling window width
    EMA_fast_label = "EMA" + str(EMA_fast_period)  # moving average column name
    EMA_fast_color = LIGHT_COLOR
    EMA_fast = ta.EMA(close, timeperiod=EMA_fast_period)

    EMA_slow_period = 200  # moving average rolling window width
    EMA_slow_label = "EMA" + str(EMA_slow_period)  # moving average column name
    EMA_slow_color = DARK_COLOR
    EMA_slow = ta.EMA(close, timeperiod=EMA_slow_period)

    # ++ Bollinger Bands ++
    BBands_period = 20
    BBands_stdev = 2
    BBands_label = (
        "BBands(" + str(BBands_period) + "," + str(BBands_stdev) + ")"
    )
    BB_upper_color = BBANDS_COLOR
    BB_avg_color = BBANDS_COLOR
    BB_lower_color = BBANDS_COLOR
    BB_upper, BB_avg, BB_lower = ta.BBANDS(
        close,
        timeperiod=BBands_period,
        nbdevup=BBands_stdev,
        nbdevdn=BBands_stdev,
        matype=0,
    )  # simple-moving-average

    # ++ On-Balance-Volume ++
    # OBV_period = 10  # OBV's EMA period
    OBV_period = 50  # OBV's EMA period
    OBV, OBV_EMA, OBV_slope = OBV_calc(
        df, symbol, EMA_pd=OBV_period, tail_pd=5, norm_pd=30
    )

    # ++ On-Balance-Volume Difference ++
    OBV_diff = OBV - OBV_EMA

    # ++ UI M.W., Ulcer-Index Moving-Window ++
    # get drawdown array
    (
        period_yr,
        CAGR,
        CAGR_Std,
        CAGR_UI,
        daily_return_std,
        Std_UI,
        drawdown,
        ulcer_index,
        max_drawdown,
    ) = symb_perf_stats(close)
    # calculate Ulcer_Index of moving-window applied to drawdown
    UI_MW_short = UI_MW(drawdown, window=window_short)
    UI_MW_long = UI_MW(drawdown, window=window_long)

    # ++ Diff. UI M.W., difference in UI_MW_short and long arrays ++
    diff_UI_MW_short = np.diff(UI_MW_short)
    diff_UI_MW_long = np.diff(UI_MW_long)
    # num of leading NaN needed to pad arrays to the same length as 'close'
    num_NA_pad_short = len(close) - len(diff_UI_MW_short)
    num_NA_pad_long = len(close) - len(diff_UI_MW_long)
    # add leading NaN pad to the arrays
    diff_UI_MW_short = np.pad(
        diff_UI_MW_short,
        (num_NA_pad_short, 0),
        "constant",
        constant_values=(np.nan),
    )
    diff_UI_MW_long = np.pad(
        diff_UI_MW_long,
        (num_NA_pad_long, 0),
        "constant",
        constant_values=(np.nan),
    )

    # +++++ Cache +++++
    # change date index from 2018-09-04 00:00:00 to 2018-09-04
    #   dtype also change from datetime64[ns] to object
    date = df.index[-1].strftime("%Y-%m-%d")
    cache = (
        symbol,
        date,
        UI_MW_short[-1],
        UI_MW_long[-1],
        diff_UI_MW_short[-1],
        diff_UI_MW_long[-1],
    )

    # +++++ Set Volume Bar Colors +++++
    colors_volume_bar = []

    for i, _ in enumerate(df.index):
        if i != 0:
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                colors_volume_bar.append(INCREASING_COLOR)
            else:
                colors_volume_bar.append(DECREASING_COLOR)
        else:
            colors_volume_bar.append(DECREASING_COLOR)

    # +++++ Set OBV_diff Bar Colors +++++
    colors_OBV_diff_bar = []

    for value in OBV_diff:
        if value > 0:
            colors_OBV_diff_bar.append(INCREASING_COLOR)
        else:
            colors_OBV_diff_bar.append(DECREASING_COLOR)

    # +++++++ panel definition ++++++++
    # ++ drawdown panel ++
    DD_title = "DD"
    DD_label = DD_title
    DD_panel_text1 = "UI: " + str("%.3f" % ulcer_index)
    DD_panel_text2 = "MaxDD: " + str("%.3f" % max_drawdown)
    DD_panel_trace1 = drawdown
    DD_trace1_color = DRAWDOWN_COLOR
    DD_trace2_color = None

    # ++ OBV panel ++
    OBV_title = "OBV"
    OBV_label = "OBV"
    OBV_EMA_label = "EMA" + str(OBV_period)
    OBV_panel_text1 = "OBV Slope: " + "%.3f" % OBV_slope
    OBV_panel_text2 = None  # NOQA
    OBV_panel_trace1 = OBV
    OBV_trace1_color = LIGHT_COLOR
    OBV_panel_trace2 = OBV_EMA
    OBV_trace2_color = DARK_COLOR

    # ++ top panel ++
    top_title = "Diff. OBV"
    top_panel_text1 = OBV_label + "-" + OBV_EMA_label
    top_panel_trace1 = OBV_diff
    top_trace1_color = LIGHT_COLOR

    # ++ middle panel ++
    mid_title = "UI MW"
    mid_panel_text1 = "MW" + str(window_short)
    mid_panel_text2 = "MW" + str(window_long)
    mid_panel_trace1 = UI_MW_short
    mid_panel_trace2 = UI_MW_long
    mid_trace1_color = LIGHT_COLOR
    mid_trace2_color = DARK_COLOR

    # ++ bottom panel ++
    btm_title = "Diff. UI M.W."
    btm_panel_text1 = "MW" + str(window_short)
    btm_panel_text2 = "MW" + str(window_long)
    btm_panel_trace1 = diff_UI_MW_short
    btm_panel_trace2 = diff_UI_MW_long
    btm_trace1_color = LIGHT_COLOR
    btm_trace2_color = DARK_COLOR

    # +++++++ plotly layout ++++++++
    layout = {
        # borders
        "margin": {
            "t": 20,
            "b": 20,
            "r": 20,
            "l": 80,
        },
        "annotations": [
            # symbol
            {
                "x": 0.25,
                "y": 0.998,
                "showarrow": False,
                "text": "<b>" + symbol + "</b>",  # bold
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {
                    "size": 15,  # larger font
                },
            },
            # period
            {
                "x": x_annotation,
                "y": y70_txt,  # 1st line
                "text": "Years: " + "%.2f" % period_yr,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # Std(Daily Returns) / Ulcer_Index, Smaller is better
            {
                "x": x_annotation,
                "y": y70_txt - 1 * y_annotation_gap,  # 2nd line
                "text": "Std/UI: " + "%.3f" % Std_UI,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # CAGR
            {
                "x": x_annotation,
                "y": y70_txt - 2 * y_annotation_gap,  # 3rd line
                "text": "CAGR: " + "%.3f" % CAGR,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # CAGR / Std
            {
                "x": x_annotation,
                "y": y70_txt - 3 * y_annotation_gap,  # 4th line
                "text": "CAGR/Std: " + "%.3f" % CAGR_Std,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # CAGR / UI
            {
                "x": x_annotation,
                "y": y70_txt - 4 * y_annotation_gap,  # 5th line
                "text": "CAGR/UI: " + "%.3f" % CAGR_UI,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # drawdown panel text1 line
            {
                "x": x_annotation,
                "y": y50_txt,
                "text": DD_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DD_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # drawdown panel text2 line
            {
                "x": x_annotation,
                "y": y50_txt - y_annotation_gap,
                "text": DD_panel_text2,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DD_trace2_color,
                "ax": -60,
                "ay": 0,
            },
            # OBV panel text1 line
            {
                "x": x_annotation,
                "y": y40_txt,
                "text": OBV_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": OBV_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # top panel text1 line
            {
                "x": x_annotation,
                "y": y30_txt,
                "text": top_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": top_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # middle panel text1 line
            {
                "x": x_annotation,
                "y": y20_txt,
                "text": mid_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": mid_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # middle panel text2 line
            {
                "x": x_annotation,
                "y": y20_txt - y_annotation_gap,
                "text": mid_panel_text2,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": mid_trace2_color,
                "ax": -60,
                "ay": 0,
            },
            # bottom panel text1
            {
                "x": x_annotation,
                "y": y10_txt,
                "text": btm_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": btm_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # bottom panel text2
            {
                "x": x_annotation,
                "y": y10_txt - y_annotation_gap,
                "text": btm_panel_text2,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": btm_trace2_color,
                "ax": -60,
                "ay": 0,
            },
        ],
        "xaxis": {
            # vertical spike line
            "showspikes": True,
            "spikemode": "toaxis+across",
            "spikedash": "solid",
            "spikethickness": 1,
            "spikecolor": spike_line_color,
            # range slider bar
            "rangeslider": {
                "visible": False  # removes rangeslider panel at bottom
            },
            # range buttons to interact
            "rangeselector": {
                "visible": True,
                "bgcolor": "rgba(150, 200, 250, 0.4)",  # background color
                "x": 0,
                "y": 1,
                "buttons": [
                    {"count": 1, "label": "reset", "step": "all"},
                    {
                        "count": 1,
                        "label": "1 yr",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "count": 6,
                        "label": "6 mo",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 3,
                        "label": "3 mo",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 1,
                        "label": "1 mo",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 1,
                        "label": "ytd",
                        "step": "year",
                        "stepmode": "todate",
                    },
                ],
            },
        },
        # panel legends
        "legend": {
            "orientation": "h",
            "y": 0.99,
            "x": 0.3,
            "yanchor": "bottom",
        },
        # panel spike line
        "yaxis80": {
            "domain": [y80_btm, y80_top],
            "showticklabels": False,
            "showline": True,
        },
        # panel price
        "yaxis70": {
            "domain": [y70_btm, y70_top],
            "showticklabels": True,
            "showline": True,
            "title": "Price",
            "type": "log",
        },
        # panel volume
        "yaxis60": {
            "domain": [y60_btm, y60_top],
            "showticklabels": True,
            "showline": True,
            "title": "Vol.",
            "titlefont": {"size": 12},
            # 'type': 'log',
        },
        # panel drawdown
        "yaxis50": {
            "domain": [y50_btm, y50_top],
            "showticklabels": True,
            "showline": True,
            "title": DD_title,
            "titlefont": {"size": 12},
            # 'type': 'log',
        },
        # panel OBV
        "yaxis40": {
            "domain": [y40_btm, y40_top],
            "showticklabels": True,
            "showline": True,
            "title": OBV_title,
            "titlefont": {"size": 12},
            # 'type': 'log',
        },
        # panel top
        "yaxis30": {
            "domain": [y30_btm, y30_top],
            "showticklabels": True,
            "showline": True,  # vertical line on right side
            "title": top_title,
            "titlefont": {"size": 12},
            "tickmode": "array",  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
            # 'tickvals': [25, 75],  # ticklines
            # 'type': 'log',
        },
        # panel mid
        "yaxis20": {
            "domain": [y20_btm, y20_top],
            "showticklabels": True,
            "showline": True,
            "title": mid_title,
            "titlefont": {"size": 12},
            "tickmode": "array",  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
            # 'type': 'log',
        },
        # panel bottom
        "yaxis10": {
            "domain": [y10_btm, y10_top],
            "showticklabels": True,
            "showline": True,
            "title": btm_title,
            "titlefont": {"size": 12},
            # 'type': 'log',
            # 'tickmode': 'array',  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
        },
    }

    # ++++++++++++ traces +++++++++++++
    data = []

    # spike line
    trace_spike_line = go.Scatter(
        # .astype(float) converts df.volume from integer to float for talib
        x=df.index,
        y=spike_line,
        yaxis="y80",
        line=dict(width=1),
        marker=dict(color=spike_line_color),
        hoverinfo="none",
        name="",
        showlegend=False,
    )
    data.append(trace_spike_line)

    # Candlestick Chart
    trace_price = go.Candlestick(
        x=df.index,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        yaxis="y70",
        name=symbol,
        increasing=dict(line=dict(color=INCREASING_COLOR)),
        decreasing=dict(line=dict(color=DECREASING_COLOR)),
        showlegend=False,  # don't show legend across top of graph
        hoverinfo="all",  # display date, OHLC
    )
    data.append(trace_price)

    # Moving Averages
    trace_EMA_fast = go.Scatter(
        x=df.index,
        y=EMA_fast,
        yaxis="y70",
        name=EMA_fast_label,
        marker=dict(color=EMA_fast_color),
        hoverinfo="all",
        line=dict(width=1),
    )
    data.append(trace_EMA_fast)

    trace_EMA_slow = go.Scatter(
        x=df.index,
        y=EMA_slow,
        yaxis="y70",
        name=EMA_slow_label,
        marker=dict(color=EMA_slow_color),
        hoverinfo="all",
        line=dict(width=1),
    )
    data.append(trace_EMA_slow)

    # Bollinger Bands
    trace_BB_upper = go.Scatter(
        x=df.index,
        y=BB_upper,
        yaxis="y70",
        line=dict(width=1),
        marker=dict(color=BB_upper_color),
        hoverinfo="all",
        name=BBands_label,
        legendgroup="Bollinger Bands",
    )
    data.append(trace_BB_upper)

    trace_BB_avg = go.Scatter(
        x=df.index,
        y=BB_avg,
        yaxis="y70",
        line=dict(width=1),
        marker=dict(color=BB_avg_color),
        hoverinfo="all",
        name=BBands_label,
        showlegend=False,  # only show legend for upperband
        legendgroup="Bollinger Bands",
    )
    data.append(trace_BB_avg)

    trace_BB_lower = go.Scatter(
        x=df.index,
        y=BB_lower,
        yaxis="y70",
        line=dict(width=1),
        marker=dict(color=BB_lower_color),
        hoverinfo="all",
        name=BBands_label,
        showlegend=False,  # only show legend for upperband
        legendgroup="Bollinger Bands",
    )
    data.append(trace_BB_lower)

    # Volume Bars
    trace_vol = go.Bar(
        x=df.index,
        y=df.volume,
        marker=dict(color=colors_volume_bar),
        yaxis="y60",
        name="Volume",
        showlegend=False,
    )
    data.append(trace_vol)

    # panel traces
    drawdown_trace1 = go.Bar(
        x=df.index,
        y=DD_panel_trace1,
        yaxis="y50",
        # line=dict(width=1),
        marker=dict(color=DD_trace1_color),
        # hoverinfo='all',
        name=DD_label,
        showlegend=False,
    )
    data.append(drawdown_trace1)

    OBV_trace1 = go.Scatter(
        x=df.index,
        y=OBV_panel_trace1,
        yaxis="y40",
        line=dict(width=1),
        marker=dict(color=OBV_trace1_color),
        # hoverinfo='all',
        name=OBV_label,
        showlegend=False,
    )
    data.append(OBV_trace1)

    OBV_trace2 = go.Scatter(
        x=df.index,
        y=OBV_panel_trace2,
        yaxis="y40",
        # line=dict(width=1),
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=OBV_trace2_color),
        # hoverinfo='all',
        name=OBV_EMA_label,
        showlegend=False,
    )
    data.append(OBV_trace2)

    top_trace1 = go.Bar(
        x=df.index,
        y=top_panel_trace1,
        yaxis="y30",
        # line=dict(width=1),
        marker=dict(color=colors_OBV_diff_bar),
        hoverinfo="all",
        name=top_panel_text1,
        showlegend=False,
    )
    data.append(top_trace1)

    mid_trace1 = go.Scatter(
        x=df.index,
        y=mid_panel_trace1,
        yaxis="y20",
        line=dict(width=1),
        marker=dict(color=mid_trace1_color),
        hoverinfo="all",
        name=mid_panel_text1,
        showlegend=False,
    )
    data.append(mid_trace1)

    mid_trace2 = go.Scatter(
        x=df.index,
        y=mid_panel_trace2,
        # yaxis='y20', line=dict(width=1),
        yaxis="y20",
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=mid_trace2_color),
        hoverinfo="all",
        name=mid_panel_text2,
        showlegend=False,
    )
    data.append(mid_trace2)

    btm_trace1 = go.Scatter(
        x=df.index,
        y=btm_panel_trace1,
        yaxis="y10",
        line=dict(width=1),
        marker=dict(color=btm_trace1_color),
        hoverinfo="all",
        name=btm_panel_text1,
        showlegend=False,
    )
    data.append(btm_trace1)

    btm_trace2 = go.Scatter(
        x=df.index,
        y=btm_panel_trace2,
        # yaxis='y10', line=dict(width=1),
        yaxis="y10",
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=btm_trace2_color),
        hoverinfo="all",
        name=btm_panel_text2,
        showlegend=False,
    )
    data.append(btm_trace2)

    if plot_chart:
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=chart_path + chart_name)

    return cache



def yf_candlestick(symbol, df, plot_chart=True):
    """Plot candlestick chart and returns:
        symbol, date, UI_MW_short[-1], UI_MW_long[-1],
        diff_UI_MW_short[-1], diff_UI_MW_long[-1]

    Args:
        symbol(string): symbol for the dataframe df
        df(dataframe): dataframe with date index
                       and columns open, high, low, close, volume
        plot_chart(bool): plots chart if True, default is True

    Return:
        cache(tuple) of:
            symbol(str): stock symbol, e.g. 'AMZN'
            date(str): plot's last date, e.g. '2018-01-05'
            UI_MW_short[-1](float): last value of UI_MW_short
            UI_MW_long[-1](float): last value of UI_MW_long
            diff_UI_MW_short[-1](float): last value of diff_UI_MW_short
            diff_UI_MW_long[-1](float): last value of diff_UI_MW_long
    """
    # 2022-08-21 change OBV's EMA period from 10 to 50, i.e.  OBV_period = 50
    # 2023-02-28 change from df.close.values to df.Close.values 


    # Reference:
    # https://www.kaggle.com/arrowsx/crypto-currency-a-plotly-sklearn-tutorial
    # https://plotly.com/python/candlestick-charts/
    # http://mrjbq7.github.io/ta-lib/
    # https://www.somesolvedproblems.com/2018/10/how-to-customize-plotlys-modebar.html
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Candlestick.html#plotly.graph_objects.Candlestick
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.html
    # https://community.plotly.com/t/random-vertical-lines-extending-to-x-axis-on-line-chart/23462
    # https://plotly.com/python/hover-text-and-formatting/

    from plotly.offline import plot
    from myUtils import symb_perf_stats, OBV_calc, UI_MW
    import plotly.graph_objs as go
    import talib as ta
    import numpy as np

    # **** IMPORTANT: COMMENT-OUT LINES INCLUDING THE BRACKETS {...} ****
    # if no lines are between the brackets, chart will be plotted compressed
    # to the right side and leaving empty spaces on the left side
    # ********

    chart_path = "C:/Users/ping/Desktop/plotly/"  # directory to store charts
    chart_name = symbol + ".html"


    # close = df.close.values  # convert dataframe to numpy array
    close = df.Close.values  # convert dataframe to numpy array


    # +++++++++ panel layout ++++++++++
    x_annotation = 0.06  # text x position
    y_annotation_gap = 0.022  # text y spacing

    # panels' y positions
    y80_top = 0.99  # spike, height 1
    y80_btm = 0.98
    y70_top = 0.98  # price, height 38
    y70_txt = 0.96  # price, text
    y70_btm = 0.60
    # gap of 1
    y60_top = 0.59  # volume, height 5
    # y60_txt = .57  # volume, text
    y60_btm = 0.54
    # gap of 1
    y50_top = 0.53  # drawdown, height 5
    y50_txt = 0.52  # drawdown, text
    y50_btm = 0.48
    # gap of 1
    y40_top = 0.47  # OBV, height 5
    y40_txt = 0.46  # OBV, text
    y40_btm = 0.42
    # gap of 1
    y30_top = 0.41  # top, height 13
    y30_txt = 0.40  # top, text
    y30_btm = 0.28
    # gap of 1
    y20_top = 0.27  # middle, height 13
    y20_txt = 0.26  # middle, text
    y20_btm = 0.14
    # gap of 1
    y10_top = 0.13  # bottom, height 13
    y10_txt = 0.12  # bottom, text
    y10_btm = 0.0

    # +++++++++ chart colors ++++++++++
    # http://colormind.io/template/paper-dashboard/#
    INCREASING_COLOR = "#17BECF"
    DECREASING_COLOR = "#7F7F7F"
    BBANDS_COLOR = "#BDBCBC"
    LIGHT_COLOR = "#0194A2"
    DARK_COLOR = "#015C65"
    DRAWDOWN_COLOR = "#5B5170"

    # ++++++++++ moving window sizes +++++++++++
    window_short = 15
    window_long = 30

    # ++++++++++ indicators +++++++++++
    # ++ spike line ++
    # With Toggle Spike Lines is on, gives vertical lines down the chart
    #   when cursor is at top of chart
    spike_line = np.zeros(len(df.index))  # create a trace with zeros
    spike_line_color = DRAWDOWN_COLOR

    # ++ moving averages ++
    EMA_fast_period = 50  # moving average rolling window width
    EMA_fast_label = "EMA" + str(EMA_fast_period)  # moving average column name
    EMA_fast_color = LIGHT_COLOR
    EMA_fast = ta.EMA(close, timeperiod=EMA_fast_period)

    EMA_slow_period = 200  # moving average rolling window width
    EMA_slow_label = "EMA" + str(EMA_slow_period)  # moving average column name
    EMA_slow_color = DARK_COLOR
    EMA_slow = ta.EMA(close, timeperiod=EMA_slow_period)

    # ++ Bollinger Bands ++
    BBands_period = 20
    BBands_stdev = 2
    BBands_label = (
        "BBands(" + str(BBands_period) + "," + str(BBands_stdev) + ")"
    )
    BB_upper_color = BBANDS_COLOR
    BB_avg_color = BBANDS_COLOR
    BB_lower_color = BBANDS_COLOR
    BB_upper, BB_avg, BB_lower = ta.BBANDS(
        close,
        timeperiod=BBands_period,
        nbdevup=BBands_stdev,
        nbdevdn=BBands_stdev,
        matype=0,
    )  # simple-moving-average

    # ++ On-Balance-Volume ++
    # OBV_period = 10  # OBV's EMA period
    OBV_period = 50  # OBV's EMA period
    OBV, OBV_EMA, OBV_slope = OBV_calc(
        df, symbol, EMA_pd=OBV_period, tail_pd=5, norm_pd=30
    )

    # ++ On-Balance-Volume Difference ++
    OBV_diff = OBV - OBV_EMA

    # ++ UI M.W., Ulcer-Index Moving-Window ++
    # get drawdown array
    (
        period_yr,
        CAGR,
        CAGR_Std,
        CAGR_UI,
        daily_return_std,
        Std_UI,
        drawdown,
        ulcer_index,
        max_drawdown,
    ) = symb_perf_stats(close)
    # calculate Ulcer_Index of moving-window applied to drawdown
    UI_MW_short = UI_MW(drawdown, window=window_short)
    UI_MW_long = UI_MW(drawdown, window=window_long)

    # ++ Diff. UI M.W., difference in UI_MW_short and long arrays ++
    diff_UI_MW_short = np.diff(UI_MW_short)
    diff_UI_MW_long = np.diff(UI_MW_long)
    # num of leading NaN needed to pad arrays to the same length as 'close'
    num_NA_pad_short = len(close) - len(diff_UI_MW_short)
    num_NA_pad_long = len(close) - len(diff_UI_MW_long)
    # add leading NaN pad to the arrays
    diff_UI_MW_short = np.pad(
        diff_UI_MW_short,
        (num_NA_pad_short, 0),
        "constant",
        constant_values=(np.nan),
    )
    diff_UI_MW_long = np.pad(
        diff_UI_MW_long,
        (num_NA_pad_long, 0),
        "constant",
        constant_values=(np.nan),
    )

    # +++++ Cache +++++
    # change date index from 2018-09-04 00:00:00 to 2018-09-04
    #   dtype also change from datetime64[ns] to object
    date = df.index[-1].strftime("%Y-%m-%d")
    cache = (
        symbol,
        date,
        UI_MW_short[-1],
        UI_MW_long[-1],
        diff_UI_MW_short[-1],
        diff_UI_MW_long[-1],
    )

    # +++++ Set Volume Bar Colors +++++
    colors_volume_bar = []

    for i, _ in enumerate(df.index):
        if i != 0:
            # if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                colors_volume_bar.append(INCREASING_COLOR)
            else:
                colors_volume_bar.append(DECREASING_COLOR)
        else:
            colors_volume_bar.append(DECREASING_COLOR)

    # +++++ Set OBV_diff Bar Colors +++++
    colors_OBV_diff_bar = []

    for value in OBV_diff:
        if value > 0:
            colors_OBV_diff_bar.append(INCREASING_COLOR)
        else:
            colors_OBV_diff_bar.append(DECREASING_COLOR)

    # +++++++ panel definition ++++++++
    # ++ drawdown panel ++
    DD_title = "DD"
    DD_label = DD_title
    DD_panel_text1 = "UI: " + str("%.3f" % ulcer_index)
    DD_panel_text2 = "MaxDD: " + str("%.3f" % max_drawdown)
    DD_panel_trace1 = drawdown
    DD_trace1_color = DRAWDOWN_COLOR
    DD_trace2_color = None

    # ++ OBV panel ++
    OBV_title = "OBV"
    OBV_label = "OBV"
    OBV_EMA_label = "EMA" + str(OBV_period)
    OBV_panel_text1 = "OBV Slope: " + "%.3f" % OBV_slope
    OBV_panel_text2 = None  # NOQA
    OBV_panel_trace1 = OBV
    OBV_trace1_color = LIGHT_COLOR
    OBV_panel_trace2 = OBV_EMA
    OBV_trace2_color = DARK_COLOR

    # ++ top panel ++
    top_title = "Diff. OBV"
    top_panel_text1 = OBV_label + "-" + OBV_EMA_label
    top_panel_trace1 = OBV_diff
    top_trace1_color = LIGHT_COLOR

    # ++ middle panel ++
    mid_title = "UI MW"
    mid_panel_text1 = "MW" + str(window_short)
    mid_panel_text2 = "MW" + str(window_long)
    mid_panel_trace1 = UI_MW_short
    mid_panel_trace2 = UI_MW_long
    mid_trace1_color = LIGHT_COLOR
    mid_trace2_color = DARK_COLOR

    # ++ bottom panel ++
    btm_title = "Diff. UI M.W."
    btm_panel_text1 = "MW" + str(window_short)
    btm_panel_text2 = "MW" + str(window_long)
    btm_panel_trace1 = diff_UI_MW_short
    btm_panel_trace2 = diff_UI_MW_long
    btm_trace1_color = LIGHT_COLOR
    btm_trace2_color = DARK_COLOR

    # +++++++ plotly layout ++++++++
    layout = {
        # borders
        "margin": {
            "t": 20,
            "b": 20,
            "r": 20,
            "l": 80,
        },
        "annotations": [
            # symbol
            {
                "x": 0.25,
                "y": 0.998,
                "showarrow": False,
                "text": "<b>" + symbol + "</b>",  # bold
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {
                    "size": 15,  # larger font
                },
            },
            # period
            {
                "x": x_annotation,
                "y": y70_txt,  # 1st line
                "text": "Years: " + "%.2f" % period_yr,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # Std(Daily Returns) / Ulcer_Index, Smaller is better
            {
                "x": x_annotation,
                "y": y70_txt - 1 * y_annotation_gap,  # 2nd line
                "text": "Std/UI: " + "%.3f" % Std_UI,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # CAGR
            {
                "x": x_annotation,
                "y": y70_txt - 2 * y_annotation_gap,  # 3rd line
                "text": "CAGR: " + "%.3f" % CAGR,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # CAGR / Std
            {
                "x": x_annotation,
                "y": y70_txt - 3 * y_annotation_gap,  # 4th line
                "text": "CAGR/Std: " + "%.3f" % CAGR_Std,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # CAGR / UI
            {
                "x": x_annotation,
                "y": y70_txt - 4 * y_annotation_gap,  # 5th line
                "text": "CAGR/UI: " + "%.3f" % CAGR_UI,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DECREASING_COLOR,
                "ax": -60,
                "ay": 0,
                "font": {
                    "size": 12,  # default font size is 12
                },
            },
            # drawdown panel text1 line
            {
                "x": x_annotation,
                "y": y50_txt,
                "text": DD_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DD_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # drawdown panel text2 line
            {
                "x": x_annotation,
                "y": y50_txt - y_annotation_gap,
                "text": DD_panel_text2,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": DD_trace2_color,
                "ax": -60,
                "ay": 0,
            },
            # OBV panel text1 line
            {
                "x": x_annotation,
                "y": y40_txt,
                "text": OBV_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": OBV_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # top panel text1 line
            {
                "x": x_annotation,
                "y": y30_txt,
                "text": top_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": top_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # middle panel text1 line
            {
                "x": x_annotation,
                "y": y20_txt,
                "text": mid_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": mid_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # middle panel text2 line
            {
                "x": x_annotation,
                "y": y20_txt - y_annotation_gap,
                "text": mid_panel_text2,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": mid_trace2_color,
                "ax": -60,
                "ay": 0,
            },
            # bottom panel text1
            {
                "x": x_annotation,
                "y": y10_txt,
                "text": btm_panel_text1,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": btm_trace1_color,
                "ax": -60,
                "ay": 0,
            },
            # bottom panel text2
            {
                "x": x_annotation,
                "y": y10_txt - y_annotation_gap,
                "text": btm_panel_text2,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": True,
                "arrowhead": 0,
                "arrowwidth": 1,
                "arrowcolor": btm_trace2_color,
                "ax": -60,
                "ay": 0,
            },
        ],
        "xaxis": {
            # vertical spike line
            "showspikes": True,
            "spikemode": "toaxis+across",
            "spikedash": "solid",
            "spikethickness": 1,
            "spikecolor": spike_line_color,
            # range slider bar
            "rangeslider": {
                "visible": False  # removes rangeslider panel at bottom
            },
            # range buttons to interact
            "rangeselector": {
                "visible": True,
                "bgcolor": "rgba(150, 200, 250, 0.4)",  # background color
                "x": 0,
                "y": 1,
                "buttons": [
                    {"count": 1, "label": "reset", "step": "all"},
                    {
                        "count": 1,
                        "label": "1 yr",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {
                        "count": 6,
                        "label": "6 mo",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 3,
                        "label": "3 mo",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 1,
                        "label": "1 mo",
                        "step": "month",
                        "stepmode": "backward",
                    },
                    {
                        "count": 1,
                        "label": "ytd",
                        "step": "year",
                        "stepmode": "todate",
                    },
                ],
            },
        },
        # panel legends
        "legend": {
            "orientation": "h",
            "y": 0.99,
            "x": 0.3,
            "yanchor": "bottom",
        },
        # panel spike line
        "yaxis80": {
            "domain": [y80_btm, y80_top],
            "showticklabels": False,
            "showline": True,
        },
        # panel price
        "yaxis70": {
            "domain": [y70_btm, y70_top],
            "showticklabels": True,
            "showline": True,
            "title": "Price",
            "type": "log",
        },
        # panel volume
        "yaxis60": {
            "domain": [y60_btm, y60_top],
            "showticklabels": True,
            "showline": True,
            "title": "Vol.",
            "titlefont": {"size": 12},
            # 'type': 'log',
        },
        # panel drawdown
        "yaxis50": {
            "domain": [y50_btm, y50_top],
            "showticklabels": True,
            "showline": True,
            "title": DD_title,
            "titlefont": {"size": 12},
            # 'type': 'log',
        },
        # panel OBV
        "yaxis40": {
            "domain": [y40_btm, y40_top],
            "showticklabels": True,
            "showline": True,
            "title": OBV_title,
            "titlefont": {"size": 12},
            # 'type': 'log',
        },
        # panel top
        "yaxis30": {
            "domain": [y30_btm, y30_top],
            "showticklabels": True,
            "showline": True,  # vertical line on right side
            "title": top_title,
            "titlefont": {"size": 12},
            "tickmode": "array",  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
            # 'tickvals': [25, 75],  # ticklines
            # 'type': 'log',
        },
        # panel mid
        "yaxis20": {
            "domain": [y20_btm, y20_top],
            "showticklabels": True,
            "showline": True,
            "title": mid_title,
            "titlefont": {"size": 12},
            "tickmode": "array",  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
            # 'type': 'log',
        },
        # panel bottom
        "yaxis10": {
            "domain": [y10_btm, y10_top],
            "showticklabels": True,
            "showline": True,
            "title": btm_title,
            "titlefont": {"size": 12},
            # 'type': 'log',
            # 'tickmode': 'array',  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
        },
    }

    # ++++++++++++ traces +++++++++++++
    data = []

    # spike line
    trace_spike_line = go.Scatter(
        # .astype(float) converts df.volume from integer to float for talib
        x=df.index,
        y=spike_line,
        yaxis="y80",
        line=dict(width=1),
        marker=dict(color=spike_line_color),
        hoverinfo="none",
        name="",
        showlegend=False,
    )
    data.append(trace_spike_line)

    # Candlestick Chart
    trace_price = go.Candlestick(
        x=df.index,
        # open=df.open,
        # high=df.high,
        # low=df.low,
        # close=df.close,
        open=df.Open,
        high=df.High,
        low=df.Low,        
        close=df.Close,    
        yaxis="y70",
        name=symbol,
        increasing=dict(line=dict(color=INCREASING_COLOR)),
        decreasing=dict(line=dict(color=DECREASING_COLOR)),
        showlegend=False,  # don't show legend across top of graph
        hoverinfo="all",  # display date, OHLC
    )
    data.append(trace_price)

    # Moving Averages
    trace_EMA_fast = go.Scatter(
        x=df.index,
        y=EMA_fast,
        yaxis="y70",
        name=EMA_fast_label,
        marker=dict(color=EMA_fast_color),
        hoverinfo="all",
        line=dict(width=1),
    )
    data.append(trace_EMA_fast)

    trace_EMA_slow = go.Scatter(
        x=df.index,
        y=EMA_slow,
        yaxis="y70",
        name=EMA_slow_label,
        marker=dict(color=EMA_slow_color),
        hoverinfo="all",
        line=dict(width=1),
    )
    data.append(trace_EMA_slow)

    # Bollinger Bands
    trace_BB_upper = go.Scatter(
        x=df.index,
        y=BB_upper,
        yaxis="y70",
        line=dict(width=1),
        marker=dict(color=BB_upper_color),
        hoverinfo="all",
        name=BBands_label,
        legendgroup="Bollinger Bands",
    )
    data.append(trace_BB_upper)

    trace_BB_avg = go.Scatter(
        x=df.index,
        y=BB_avg,
        yaxis="y70",
        line=dict(width=1),
        marker=dict(color=BB_avg_color),
        hoverinfo="all",
        name=BBands_label,
        showlegend=False,  # only show legend for upperband
        legendgroup="Bollinger Bands",
    )
    data.append(trace_BB_avg)

    trace_BB_lower = go.Scatter(
        x=df.index,
        y=BB_lower,
        yaxis="y70",
        line=dict(width=1),
        marker=dict(color=BB_lower_color),
        hoverinfo="all",
        name=BBands_label,
        showlegend=False,  # only show legend for upperband
        legendgroup="Bollinger Bands",
    )
    data.append(trace_BB_lower)

    # Volume Bars
    trace_vol = go.Bar(
        x=df.index,
        # y=df.volume,
        y=df.Volume,        
        marker=dict(color=colors_volume_bar),
        yaxis="y60",
        name="Volume",
        showlegend=False,
    )
    data.append(trace_vol)

    # panel traces
    drawdown_trace1 = go.Bar(
        x=df.index,
        y=DD_panel_trace1,
        yaxis="y50",
        # line=dict(width=1),
        marker=dict(color=DD_trace1_color),
        # hoverinfo='all',
        name=DD_label,
        showlegend=False,
    )
    data.append(drawdown_trace1)

    OBV_trace1 = go.Scatter(
        x=df.index,
        y=OBV_panel_trace1,
        yaxis="y40",
        line=dict(width=1),
        marker=dict(color=OBV_trace1_color),
        # hoverinfo='all',
        name=OBV_label,
        showlegend=False,
    )
    data.append(OBV_trace1)

    OBV_trace2 = go.Scatter(
        x=df.index,
        y=OBV_panel_trace2,
        yaxis="y40",
        # line=dict(width=1),
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=OBV_trace2_color),
        # hoverinfo='all',
        name=OBV_EMA_label,
        showlegend=False,
    )
    data.append(OBV_trace2)

    top_trace1 = go.Bar(
        x=df.index,
        y=top_panel_trace1,
        yaxis="y30",
        # line=dict(width=1),
        marker=dict(color=colors_OBV_diff_bar),
        hoverinfo="all",
        name=top_panel_text1,
        showlegend=False,
    )
    data.append(top_trace1)

    mid_trace1 = go.Scatter(
        x=df.index,
        y=mid_panel_trace1,
        yaxis="y20",
        line=dict(width=1),
        marker=dict(color=mid_trace1_color),
        hoverinfo="all",
        name=mid_panel_text1,
        showlegend=False,
    )
    data.append(mid_trace1)

    mid_trace2 = go.Scatter(
        x=df.index,
        y=mid_panel_trace2,
        # yaxis='y20', line=dict(width=1),
        yaxis="y20",
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=mid_trace2_color),
        hoverinfo="all",
        name=mid_panel_text2,
        showlegend=False,
    )
    data.append(mid_trace2)

    btm_trace1 = go.Scatter(
        x=df.index,
        y=btm_panel_trace1,
        yaxis="y10",
        line=dict(width=1),
        marker=dict(color=btm_trace1_color),
        hoverinfo="all",
        name=btm_panel_text1,
        showlegend=False,
    )
    data.append(btm_trace1)

    btm_trace2 = go.Scatter(
        x=df.index,
        y=btm_panel_trace2,
        # yaxis='y10', line=dict(width=1),
        yaxis="y10",
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=btm_trace2_color),
        hoverinfo="all",
        name=btm_panel_text2,
        showlegend=False,
    )
    data.append(btm_trace2)

    if plot_chart:
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=chart_path + chart_name)

    return cache




def yf_plot_symbols(symbols, path_data_dump, filename_pickled_df_OHLCV, date_start_limit=None,
                     date_end_limit=None, iloc_offset=None):
    """Plots symbol in symbols list. yf_download_AdjOHLCV created and pickled
    dataframe df_OHLCV which has symbols' OHLCV data. The dataframe is stored
    in directory path_data_dump. This program plots the symbols using data from
    df_OHLCV.

    (Copied from def dates_within_limits)
    Order of hierarchy to determine Start and End dates:
    1. given date if it is not None
    2. iloc_offset if it is not None

    start = date_start_limit
    end = date_end_limit
    date_start_limit sets the upper (i.e. oldest) limit.
    date_end_limit sets the lower (i.e. newest) limit.
    date_start is the oldest index date that is within the limits.
    date_end is the newest index date that is within the limits.

    START   END    ILOC-OFFSET  Return->  START DATE             END DATE
    ---------------------------------------------------------------------------
    start   end	   iloc_offset	          date_start	         date_end
    start   end	   none                   date_start	         date_end
    start   none   iloc_offset	          date_start	         date_start+iloc_offset  # NOQA
    start   none   none                   date_start	         index[-1]
    none	end	   iloc_offset	          date_end-iloc_offset   date_end
    none	end	   none                   index[0]	             date_end
    none	none   iloc_offset	          index[0]	             index[-1]
    none	none   none                   index[0]	             index[-1]

    Args:
        symbols(list[str]): list of symbols, e.g. ['AAPL', 'GOOGL', ...]
        path_symbol_data(str): directory path of pickled df_OHLCV
        date_start_limit(str): limit on the start date in format 'yyyy-mm-dd'
        date_end_limit(str): limit on the end date in format 'yyyy-mm-dd'
        iloc_offset(int): index offset
    
    Return:
        cache_returned_by_plot:
            symbol(str): stock symbol, e.g. 'AMZN'
            date(str): date, e.g. '2018-01-05'
            UI_MW_short[-1](float): last value of UI_MW_short,
                Ulcer-Index / Moving-Window-Short
            UI_MW_long[-1](float): last value of UI_MW_long
                Ulcer-Index / Moving-Window-Long
            diff_UI_MW_short[-1](float): last value of diff_UI_MW_short
                difference of last two 'Ulcer-Index / Moving-Window-Short'
            diff_UI_MW_long[-1](float): last value of diff_UI_MW_long
                difference of last two 'Ulcer-Index / Moving-Window-Long'
    """


    from myUtils import pickle_load, yf_candlestick


    print('+'*15 + '  yf_plot_symbols(symbols, filename_pickled_df_OHLCV, path_data_dump)  ' + '+'*15 + '\n')

    # directories and file names
    df_filename = filename_pickled_df_OHLCV
    # list of cache. Stores cache returned by candlestick plot
    caches_returned_by_plot = []
    # read df
    df_OHLCV = pickle_load(path_data_dump, df_filename)
    date_index_all_dates = df_OHLCV.index
    # get iloc of date_end_limit in XOM's date index
    date_start, date_end, iloc_date_start, iloc_date_end = \
        dates_within_limits(date_index_all_dates, date_start_limit,
                            date_end_limit, iloc_offset)
    df_OHLCV = df_OHLCV[date_start:date_end]
    print(f'df_OHLCV date index: {date_start} - {date_end}\n')

    for symbol in symbols:
        df = df_OHLCV[symbol].copy()
        # drop any row with NaN from df
        df.dropna(inplace=True)
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        print(f'After dropping all NaN, plot symbol {symbol} from {first_date} to {last_date}\n')   

        # cache_returned_by_plot = candlestick(symbol, df, plot_chart=False)
        cache_returned_by_plot = yf_candlestick(symbol, df, plot_chart=True)
        caches_returned_by_plot.append(cache_returned_by_plot)

    print('{}\n'.format('-'*78))

    return caches_returned_by_plot

def yf_print_symbol_data(symbols):
    """Prints symbol data from yfinance yf.Ticker(symbol).info,
    and OHLCV data for the last 5 day.

    Args:
        symbols([str]): list of symbols, e.g. ['AAPL', 'GOOGL', ...]

    Return:
        symbols_stock(list): list of stock symbols
        symbols_etf(list): list of etf symbols
        symbols_cryto(list): list of cryto symbols
    """

    import yfinance as yf
    import textwrap
    import pandas as pd
    from datetime import datetime

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 10)

    # Wrapper for longBusinessSummary
    wrapper = textwrap.TextWrapper(width=60)

    # stock/equity dict keys for obj.info
    eqKeys = [
        "symbol",
        "quoteType",
        "longName",
        "industry",
        "sector",
        "marketCap",
        "revenueGrowth",
        "earningsGrowth",
        "earningsQuarterlyGrowth",
        "ebitdaMargins",
        "previousClose",
        "fiftyTwoWeekHigh",
        "52WeekChange",
        "revenuePerShare",
        "forwardEps",
        "trailingEps",
        "sharesOutstanding",
        "longBusinessSummary",
    ]
    # ETF dict keys for obj.info
    etfKeys = [
        "symbol",
        "quoteType",
        "longName",
        "category",
        "totalAssets",
        "beta3Year",
        "navPrice",
        "previousClose",
        "holdings",
        "sectorWeightings",
        "longBusinessSummary",
    ]
    # cryto dict keys for obj.info
    crytoKeys = [
        "symbol",
        "quoteType",
        "name",
        "regularMarketPrice",
        "volume",
        "marketCap",
        "circulatingSupply",
        "startDate",
    ]

    symbols_stock = []  # symbols that are stocks
    symbols_etf = []  # symbols that are ETFs
    symbols_cryto = []  # symbols that are crytos

    for symbol in symbols:
        print("=" * 60)
        # print symbol OHLCV for the last 5 days
        print(f'{symbol}\n{yf.Ticker(symbol).history(period="5d")}\n')
        obj = yf.Ticker(symbol)  # dir(obj) lists methods in obj
        # obj.info is a dict (e.g {'exchange': 'PCX', ... ,  'logo_url': ''})
        if obj.info["quoteType"] == "EQUITY":  # its a stock
            symbols_stock.append(symbol)
            for key in eqKeys:
                value = obj.info[key]
                if key == "marketCap":
                    print(
                        f"{key:26}{value/1e9:>10,.3f}B"
                    )  # reformat to billions
                elif key == "sharesOutstanding":
                    print(
                        f"{key:26}{value/1e6:>10,.3f}M"
                    )  # reformat to millions

                elif key == "longBusinessSummary":
                    string = wrapper.fill(text=value)
                    print(f"\n{key}:\n{string}")                    
                else:
                    # if type(value) == str:  # its a string
                    if type(value) == str or value == None:  # its a string 
                        print(f"{key:26}{value}")
                    else:  # format as a number
                        print(f"{key:26}{value:>10.3f}")
            print("")
        elif obj.info["quoteType"] == "ETF":  # its an ETF
            symbols_etf.append(symbol)
            for key in etfKeys:
                value = obj.info[key]
                # obj.info['holding'] is a list of of dict
                #   e.g. [{'symbol': 'AAPL', 'holdingName': 'Apple Inc', 'holdingPercent': 0.2006}, ...]
                if key == "holdings":  # print ETF stock holdings
                    keyItems = len(value)
                    hd_heading = (
                        f"\n{'symbol':10}{'holding-percent':20}{'name'}"
                    )
                    print(hd_heading)
                    for i in range(keyItems):
                        hd_symbol = value[i]["symbol"]
                        hd_pct = value[i]["holdingPercent"]
                        hd_name = value[i]["holdingName"]
                        hd_info = f"{hd_symbol:<10}{hd_pct:<20.3f}{hd_name}"
                        print(hd_info)
                # obj.info['sectorWeightings'] is a list of of dict, similar to obj.info['holding']
                elif key == "sectorWeightings":  # print ETF sector weightings
                    keyItems = len(value)
                    hd_heading = f"\n{'sector':30}{'holding-percent':20}"
                    print(hd_heading)
                    sectors_list = value
                    for sector_dict in sectors_list:
                        for k, v in sector_dict.items():
                            print(f"{k:30}{v:<20.3f}")
                elif key == "totalAssets":
                    print(f"{key:30}{value/1e9:<.3f}B")  # reformat to billions
                elif key == "longBusinessSummary":
                    string = wrapper.fill(text=value)
                    print(f"\n{key}:\n{string}")    
                else:
                    print(f"{key:30}{value}")
            print("")
        elif obj.info["quoteType"] == "CRYPTOCURRENCY":
            symbols_cryto.append(symbol)
            for key in crytoKeys:
                value = obj.info[key]
                if key == "marketCap" or key == "volume":
                    print(
                        f"{key:26}{value/1e9:>10,.3f}B"
                    )  # reformat to billions
                elif key == "circulatingSupply":
                    print(
                        f"{key:26}{value/1e6:>10,.3f}M"
                    )  # reformat to millions
                elif key == "startDate":
                    UTC_timestamp_sec = obj.info[
                        "startDate"
                    ]  # Unix time stamp (i.e. seconds since 1970-01-01)
                    # convert Unix UTC_timestamp_sec in sec. to yyyy-mm-dd,  'startDate': 1367107200
                    startDate = datetime.fromtimestamp(
                        UTC_timestamp_sec
                    ).strftime("%Y-%m-%d")
                    print(f"{key:26}{startDate}")  # reformat to billions
                else:
                    if type(value) == str:
                        print(f"{key:26}{value}")
                    else:
                        print(f"{key:26}{value:>10,.0f}")
            print("")
        else:
            print(f'{symbol} is {obj.info["quoteType"]}')
            print("")

    return symbols_stock, symbols_etf, symbols_cryto

def chunked_list(a_list, chunk_size):
    """Split python a_list into chunks_size lists.
    e.g. a_list = [1, 2, 3, 4, 5, 6, 7], chunk_size = 2
    return chunked_list = [[1,2], [3,4], [5,6], [7]]

    Args:
        a_list(list): a list, e.g. ['AAPL', 'GOOGL', ...]
        chuck_size[int]: number of elements in the sub-list

    Return:
        chunked_list(list): list of sub-lists, where len(sub-list) == chuck_size
    """
    # Split a Python List into Chunks using For Loops
    chunked_list = []
    for i in range(0, len(a_list), chunk_size):
        chunked_list.append(a_list[i:i+chunk_size])
    return chunked_list

def delete_all_rows(path_data_dump, filename_pickle, verbose=False):
    """Load pickled df, delete all rows in df, and save pickled df.

    Args:
        path_data_dump(str): path to write the pickle file
        filename_pickle(str): pickle filename

    Return:
        None
    """  
  
    from myUtils import pickle_dump, pickle_load

    df = pickle_load(path_data_dump, filename_pickle)
    if verbose:
        print(f'{filename_pickle}.tail(3) before delete_all_rows:\n{df.tail(3)}\n\n')
    # delete all rows in df
    df = df[0:0]
    pickle_dump(df, path_data_dump, filename_pickle)
    if verbose:
        print(f'{filename_pickle}.tail(3) after delete_all_rows:\n{df.tail(3)}\n')
        print(f'{filename_pickle} save to: {path_data_dump}{filename_pickle}\n\n')
    return None
