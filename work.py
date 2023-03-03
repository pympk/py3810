def yf_plot_symbols(symbols, path_data_dump, filename_pickled_df_OHLCV, date_start_limit=None,
                    date_end_limit=None, iloc_offset=None):
    """Plots symbol in symbols list. Program ps1n2 created dictionary
    df_symbols_OHLCV_download which has symbols' OHLCV data.
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

    
    
    # from myUtils import pickle_load, dates_within_limits
    # from yf_CStick_DD_OBV_UIMW_Diff_UIMW_cache import yf_candlestick
    from myUtils import pickle_load


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

    for symbol in symbols:
        # df = df_OHLCV[symbol]
        df = df_OHLCV[date_start:date_end]
        print('plot symbol {} from {} to {}\n'
              .format(symbol, date_start, date_end))
        # cache_returned_by_plot = candlestick(symbol, df, plot_chart=False)
        cache_returned_by_plot = yf_candlestick(symbol, df, plot_chart=True)
        caches_returned_by_plot.append(cache_returned_by_plot)

    print('{}\n'.format('-'*78))

    return caches_returned_by_plot

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



    # # import pandas as pd
    # from myUtils import valid_start_end_dates, dates_adjacent_pivot



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

def yf_candlestick(symbol, df, plot_chart=True):
    """Plot candlestick chart and returns:
        symbol, date, UI_MW_short[-1], UI_MW_long[-1],
        diff_UI_MW_short[-1], diff_UI_MW_long[-1]

    Args:
        symbol(string): symbol for the dataframe df
        df(dataframe): dataframe with date index
                       and columns open, high, low, close, volume
        plot_chart(bool): plots chart if True, defaut is True

    Return:
        symbol(str): stock symbol, e.g. 'AMZN'
        date(str): date, e.g. '2018-01-05'
        UI_MW_short[-1](float): last value of UI_MW_short
        UI_MW_long[-1](float): last value of UI_MW_long
        diff_UI_MW_short[-1](float): last value of diff_UI_MW_short
        diff_UI_MW_long[-1](float): last value of diff_UI_MW_long
    """

    # Reference:
    # https://www.kaggle.com/arrowsx/crypto-currency-a-plotly-sklearn-tutorial
    # http://mrjbq7.github.io/ta-lib/

    from plotly.offline import plot
    from myUtils import symb_perf_stats, OBV_calc, UI_MW
    import plotly.graph_objs as go
    import talib as ta
    import numpy as np

    # **** IMPORTANT: COMMENT-OUT LINES INCLUDING THE BRACKETS {...} ****
    # if no lines are between the brackets, chart will be plotted compressed
    # to the right side and leaving empty spaces on the left side
    # ********

    # chart_path = 'C:/Users/ping/Desktop/plotly/'  # directory to store charts
    # chart_path = 'C:/Users/ping/OneDrive/Desktop/plotly/'  # directory to store charts
    chart_path = 'C:/Users/ping/Desktop/plotly/'  # directory to store charts
    chart_name = symbol + '.html'

    close = df.close.values  # convert dataframe to numpy array

    # +++++++++ panel layout ++++++++++
    x_annotation = 0.06  # text x position
    y_annotation_gap = 0.022  # text y spacing

    # panels' y positions
    y80_top = .99  # spike, height 1
    y80_btm = .98
    y70_top = .98  # price, height 38
    y70_txt = .96  # price, text
    y70_btm = .60
    # gap of 1
    y60_top = .59  # volume, height 5
    # y60_txt = .57  # volume, text
    y60_btm = .54
    # gap of 1
    y50_top = .53  # drawdown, height 5
    y50_txt = .52  # drawdown, text
    y50_btm = .48
    # gap of 1
    y40_top = .47  # OBV, height 5
    y40_txt = .46  # OBV, text
    y40_btm = .42
    # gap of 1
    y30_top = .41  # top, height 13
    y30_txt = .40  # top, text
    y30_btm = .28
    # gap of 1
    y20_top = .27  # middle, height 13
    y20_txt = .26  # middle, text
    y20_btm = .14
    # gap of 1
    y10_top = .13  # bottom, height 13
    y10_txt = .12  # bottom, text
    y10_btm = .0

    # +++++++++ chart colors ++++++++++
    # http://colormind.io/template/paper-dashboard/#
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'
    BBANDS_COLOR = '#BDBCBC'
    LIGHT_COLOR = '#0194A2'
    DARK_COLOR = '#015C65'
    DRAWDOWN_COLOR = '#5B5170'

    # ++++++++++ moving window sizes +++++++++++
    window_short = 15
    window_long = 30

    # ++++++++++ indicators +++++++++++
    # ++ spike line ++
    # With Toggle Spike Lines is on, gives vertical lines down the chart
    #   when cursor is at top of chart
    spike_line = np.zeros(len(df.index))  # create a trace with zeros
    spike_line_color = BBANDS_COLOR

    # ++ moving averages ++
    EMA_fast_period = 50  # moving average rolling window width
    EMA_fast_label = 'EMA' + str(EMA_fast_period)  # moving average column name
    EMA_fast_color = LIGHT_COLOR
    EMA_fast = ta.EMA(close, timeperiod=EMA_fast_period)

    EMA_slow_period = 200  # moving average rolling window width
    EMA_slow_label = 'EMA' + str(EMA_slow_period)  # moving average column name
    EMA_slow_color = DARK_COLOR
    EMA_slow = ta.EMA(close, timeperiod=EMA_slow_period)

    # ++ Bollinger Bands ++
    BBands_period = 20
    BBands_stdev = 2
    BBands_label = \
        'BBands(' + str(BBands_period) + ',' + str(BBands_stdev) + ')'
    BB_upper_color = BBANDS_COLOR
    BB_avg_color = BBANDS_COLOR
    BB_lower_color = BBANDS_COLOR
    BB_upper, BB_avg, BB_lower = ta.BBANDS(close,
                                           timeperiod=BBands_period,
                                           nbdevup=BBands_stdev,
                                           nbdevdn=BBands_stdev,
                                           matype=0)  # simple-moving-average

    # ++ On-Balance-Volume ++
    OBV_period = 10  # OBV's EMA period
    OBV, OBV_EMA, OBV_slope = OBV_calc(df, symbol, EMA_pd=OBV_period,
                                       tail_pd=5, norm_pd=30)

    # ++ On-Balance-Volume Difference ++
    OBV_diff = OBV - OBV_EMA

    # ++ UI M.W., Ulcer-Index Moving-Window ++
    # get drawdown array
    period_yr, CAGR, CAGR_Std, CAGR_UI, daily_return_std,  Std_UI, \
        drawdown, ulcer_index, max_drawdown = symb_perf_stats(close)
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
    diff_UI_MW_short = np.pad(diff_UI_MW_short, (num_NA_pad_short, 0),
                              'constant', constant_values=(np.nan))
    diff_UI_MW_long = np.pad(diff_UI_MW_long, (num_NA_pad_long, 0),
                             'constant', constant_values=(np.nan))

    # +++++ Cache +++++
    # change date index from 2018-09-04 00:00:00 to 2018-09-04
    #   dtype also change from datetime64[ns] to object
    date = df.index[-1].strftime("%Y-%m-%d")
    cache = (symbol, date, UI_MW_short[-1], UI_MW_long[-1],
             diff_UI_MW_short[-1], diff_UI_MW_long[-1])

    # +++++ Set Volume Bar Colors +++++
    colors_volume_bar = []

    for i, _ in enumerate(df.index):
        if i != 0:
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
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
    DD_title = 'DD'
    DD_label = DD_title
    DD_panel_text1 = 'UI: ' + str('%.3f' % ulcer_index)
    DD_panel_text2 = 'MaxDD: ' + str('%.3f' % max_drawdown)
    DD_panel_trace1 = drawdown
    DD_trace1_color = DRAWDOWN_COLOR
    DD_trace2_color = None

    # ++ OBV panel ++
    OBV_title = 'OBV'
    OBV_label = 'OBV'
    OBV_EMA_label = 'EMA' + str(OBV_period)
    OBV_panel_text1 = 'OBV Slope: ' + '%.3f' % OBV_slope
    OBV_panel_text2 = None  # NOQA
    OBV_panel_trace1 = OBV
    OBV_trace1_color = LIGHT_COLOR
    OBV_panel_trace2 = OBV_EMA
    OBV_trace2_color = DARK_COLOR

    # ++ top panel ++
    top_title = 'Diff. OBV'
    top_panel_text1 = OBV_label + '-' + OBV_EMA_label
    top_panel_trace1 = OBV_diff
    top_trace1_color = LIGHT_COLOR

    # ++ middle panel ++
    mid_title = 'UI MW'
    mid_panel_text1 = 'MW' + str(window_short)
    mid_panel_text2 = 'MW' + str(window_long)
    mid_panel_trace1 = UI_MW_short
    mid_panel_trace2 = UI_MW_long
    mid_trace1_color = LIGHT_COLOR
    mid_trace2_color = DARK_COLOR

    # ++ bottom panel ++
    btm_title = 'Diff. UI M.W.'
    btm_panel_text1 = 'MW' + str(window_short)
    btm_panel_text2 = 'MW' + str(window_long)
    btm_panel_trace1 = diff_UI_MW_short
    btm_panel_trace2 = diff_UI_MW_long
    btm_trace1_color = LIGHT_COLOR
    btm_trace2_color = DARK_COLOR

    # +++++++ plotly layout ++++++++
    layout = {
        # borders
        'margin': {
            't': 20,
            'b': 20,
            'r': 20,
            'l': 80,
        },
        'annotations': [
            # symbol
            {
                'x': 0.25,
                'y': 0.998,
                'showarrow': False,
                'text': '<b>' + symbol + '</b>',  # bold
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'bottom',
                'font': {
                    'size': 15,  # larger font
                }
            },
            # period
            {
                'x': x_annotation,
                'y': y70_txt,  # 1st line
                'text': 'Years: ' + '%.2f' % period_yr,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DECREASING_COLOR,
                'ax': -60,
                'ay': 0,
                'font': {
                    'size': 12,  # default font size is 12
                }
            },
            # Std(Daily Returns) / Ulcer_Index, Smaller is better
            {
                'x': x_annotation,
                'y': y70_txt - 1 * y_annotation_gap,  # 2nd line
                'text': 'Std/UI: ' + '%.3f' % Std_UI,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DECREASING_COLOR,
                'ax': -60,
                'ay': 0,
                'font': {
                    'size': 12,  # default font size is 12
                }
            },
            # CAGR
            {
                'x': x_annotation,
                'y': y70_txt - 2 * y_annotation_gap,  # 3rd line
                'text': 'CAGR: ' + '%.3f' % CAGR,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DECREASING_COLOR,
                'ax': -60,
                'ay': 0,
                'font': {
                    'size': 12,  # default font size is 12
                }
            },
            # CAGR / Std
            {
                'x': x_annotation,
                'y': y70_txt - 3 * y_annotation_gap,  # 4th line
                'text': 'CAGR/Std: ' + '%.3f' % CAGR_Std,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DECREASING_COLOR,
                'ax': -60,
                'ay': 0,
                'font': {
                    'size': 12,  # default font size is 12
                }
            },
            # CAGR / UI
            {
                'x': x_annotation,
                'y': y70_txt - 4 * y_annotation_gap,  # 5th line
                'text': 'CAGR/UI: ' + '%.3f' % CAGR_UI,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DECREASING_COLOR,
                'ax': -60,
                'ay': 0,
                'font': {
                    'size': 12,  # default font size is 12
                }
            },
            # drawdown panel text1 line
            {
                'x': x_annotation,
                'y': y50_txt,
                'text': DD_panel_text1,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DD_trace1_color,
                'ax': -60,
                'ay': 0,
            },
            # drawdown panel text2 line
            {
                'x': x_annotation,
                'y': y50_txt - y_annotation_gap,
                'text': DD_panel_text2,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': DD_trace2_color,
                'ax': -60,
                'ay': 0,
            },
            # OBV panel text1 line
            {
                'x': x_annotation,
                'y': y40_txt,
                'text': OBV_panel_text1,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': OBV_trace1_color,
                'ax': -60,
                'ay': 0,
            },
            # top panel text1 line
            {
                'x': x_annotation,
                'y': y30_txt,
                'text': top_panel_text1,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': top_trace1_color,
                'ax': -60,
                'ay': 0,
            },
            # middle panel text1 line
            {
                'x': x_annotation,
                'y': y20_txt,
                'text': mid_panel_text1,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': mid_trace1_color,
                'ax': -60,
                'ay': 0,
            },
            # middle panel text2 line
            {
                'x': x_annotation,
                'y': y20_txt - y_annotation_gap,
                'text': mid_panel_text2,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': mid_trace2_color,
                'ax': -60,
                'ay': 0,
            },
            # bottom panel text1
            {
                'x': x_annotation,
                'y': y10_txt,
                'text': btm_panel_text1,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': btm_trace1_color,
                'ax': -60,
                'ay': 0,
            },
            # bottom panel text2
            {
                'x': x_annotation,
                'y': y10_txt - y_annotation_gap,
                'text': btm_panel_text2,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'left',
                'yanchor': 'middle',
                'showarrow': True,
                'arrowhead': 0,
                'arrowwidth': 1,
                'arrowcolor': btm_trace2_color,
                'ax': -60,
                'ay': 0,
            },
        ],
        'xaxis': {
            # range slider bar
            'rangeslider': {
                'visible': False  # removes rangeslider panel at bottom
            },
            # range buttons to interact
            'rangeselector': {
                'visible': True,
                'bgcolor': 'rgba(150, 200, 250, 0.4)',  # background color
                'x': 0,
                'y': 1,
                'buttons': [
                    {'count': 1, 'label': 'reset', 'step': 'all'},
                    {'count': 1, 'label': '1 yr', 'step': 'year',
                        'stepmode': 'backward'},
                    {'count': 6, 'label': '6 mo', 'step': 'month',
                        'stepmode': 'backward'},
                    {'count': 3, 'label': '3 mo', 'step': 'month',
                        'stepmode': 'backward'},
                    {'count': 1, 'label': '1 mo', 'step': 'month',
                        'stepmode': 'backward'},
                    {'count': 1, 'label': 'ytd', 'step': 'year',
                        'stepmode': 'todate'},
                ]
            }
        },
        # panel legends
        'legend': {
            'orientation': 'h',
            'y': 0.99, 'x': 0.3,
            'yanchor': 'bottom'
        },
        # panel spike line
        'yaxis80': {
            'domain': [y80_btm, y80_top],
            'showticklabels': False,
            'showline': False,
        },
        # panel price
        'yaxis70': {
            'domain': [y70_btm, y70_top],
            'showticklabels': True,
            'showline': True,
            'title': 'Price',
            'type': 'log',
        },
        # panel volume
        'yaxis60': {
            'domain': [y60_btm, y60_top],
            'showticklabels': True,
            'showline': True,
            'title': 'Vol.',
            'titlefont': {
                'size': 12
            },
            # 'type': 'log',
        },
        # panel drawdown
        'yaxis50': {
            'domain': [y50_btm, y50_top],
            'showticklabels': True,
            'showline': True,
            'title': DD_title,
            'titlefont': {
                'size': 12
            },
            # 'type': 'log',
        },
        # panel OBV
        'yaxis40': {
            'domain': [y40_btm, y40_top],
            'showticklabels': True,
            'showline': True,
            'title': OBV_title,
            'titlefont': {
                'size': 12
            },
            # 'type': 'log',
        },
        # panel top
        'yaxis30': {
            'domain': [y30_btm, y30_top],
            'showticklabels': True,
            'showline': True,  # vertical line on right side
            'title': top_title,
            'titlefont': {
                'size': 12
            },
            'tickmode': 'array',  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
            # 'tickvals': [25, 75],  # ticklines
            # 'type': 'log',
        },
        # panel mid
        'yaxis20': {
            'domain': [y20_btm, y20_top],
            'showticklabels': True,
            'showline': True,
            'title': mid_title,
            'titlefont': {
                'size': 12
            },
            'tickmode': 'array',  # set ticklines per tickvals array
            # 'tickvals': [0, 25, 75, 100],  # ticklines
            # 'type': 'log',
        },
        # panel bottom
        'yaxis10': {
            'domain': [y10_btm, y10_top],
            'showticklabels': True,
            'showline': True,
            'title': btm_title,
            'titlefont': {
                'size': 12
            },
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
        yaxis='y80', line=dict(width=1),
        marker=dict(color=spike_line_color), hoverinfo='none',
        name='', showlegend=False,
    )
    data.append(trace_spike_line)

    # Candlestick Chart
    trace_price = go.Candlestick(
        x=df.index, open=df.open, high=df.high, low=df.low, close=df.close,
        yaxis='y70', name=symbol,
        increasing=dict(line=dict(color=INCREASING_COLOR)),
        decreasing=dict(line=dict(color=DECREASING_COLOR)),
        showlegend=False  # don't show legend across top of graph
    )
    data.append(trace_price)

    # Moving Averages
    trace_EMA_fast = go.Scatter(
        x=df.index,
        y=EMA_fast,
        yaxis='y70',
        name=EMA_fast_label,
        marker=dict(color=EMA_fast_color),
        hoverinfo='all',
        line=dict(width=1)
    )
    data.append(trace_EMA_fast)

    trace_EMA_slow = go.Scatter(
        x=df.index,
        y=EMA_slow,
        yaxis='y70',
        name=EMA_slow_label,
        marker=dict(color=EMA_slow_color),
        hoverinfo='all',
        line=dict(width=1)
    )
    data.append(trace_EMA_slow)

    # Bollinger Bands
    trace_BB_upper = go.Scatter(
        x=df.index, y=BB_upper,
        yaxis='y70', line=dict(width=1),
        marker=dict(color=BB_upper_color), hoverinfo='all',
        name=BBands_label,
        legendgroup='Bollinger Bands'
    )
    data.append(trace_BB_upper)

    trace_BB_avg = go.Scatter(
        x=df.index, y=BB_avg,
        yaxis='y70', line=dict(width=1),
        marker=dict(color=BB_avg_color), hoverinfo='all',
        name=BBands_label, showlegend=False,  # only show legend for upperband
        legendgroup='Bollinger Bands'
    )
    data.append(trace_BB_avg)

    trace_BB_lower = go.Scatter(
        x=df.index, y=BB_lower,
        yaxis='y70', line=dict(width=1),
        marker=dict(color=BB_lower_color), hoverinfo='all',
        name=BBands_label, showlegend=False,  # only show legend for upperband
        legendgroup='Bollinger Bands'
    )
    data.append(trace_BB_lower)

    # Volume Bars
    trace_vol = go.Bar(
        x=df.index, y=df.volume,
        marker=dict(color=colors_volume_bar),
        yaxis='y60', name='Volume',
        showlegend=False
    )
    data.append(trace_vol)

    # panel traces
    drawdown_trace1 = go.Bar(
        x=df.index, y=DD_panel_trace1,
        yaxis='y50',
        # line=dict(width=1),
        marker=dict(color=DD_trace1_color),
        # hoverinfo='all',
        name=DD_label, showlegend=False,
    )
    data.append(drawdown_trace1)

    OBV_trace1 = go.Scatter(
        x=df.index, y=OBV_panel_trace1,
        yaxis='y40',
        line=dict(width=1),
        marker=dict(color=OBV_trace1_color),
        # hoverinfo='all',
        name=OBV_label, showlegend=False,
    )
    data.append(OBV_trace1)

    OBV_trace2 = go.Scatter(
        x=df.index, y=OBV_panel_trace2,
        yaxis='y40',
        # line=dict(width=1),
        line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=OBV_trace2_color),
        # hoverinfo='all',
        name=OBV_EMA_label, showlegend=False,
    )
    data.append(OBV_trace2)

    top_trace1 = go.Bar(
        x=df.index, y=top_panel_trace1,
        yaxis='y30',
        # line=dict(width=1),
        marker=dict(color=colors_OBV_diff_bar), hoverinfo='all',
        name=top_panel_text1, showlegend=False,
    )
    data.append(top_trace1)

    mid_trace1 = go.Scatter(
        x=df.index, y=mid_panel_trace1,
        yaxis='y20', line=dict(width=1),
        marker=dict(color=mid_trace1_color), hoverinfo='all',
        name=mid_panel_text1, showlegend=False,
    )
    data.append(mid_trace1)

    mid_trace2 = go.Scatter(
        x=df.index, y=mid_panel_trace2,
        # yaxis='y20', line=dict(width=1),
        yaxis='y20', line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=mid_trace2_color), hoverinfo='all',
        name=mid_panel_text2, showlegend=False,
    )
    data.append(mid_trace2)

    btm_trace1 = go.Scatter(
        x=df.index, y=btm_panel_trace1,
        yaxis='y10', line=dict(width=1),
        marker=dict(color=btm_trace1_color), hoverinfo='all',
        name=btm_panel_text1, showlegend=False,
    )
    data.append(btm_trace1)

    btm_trace2 = go.Scatter(
        x=df.index, y=btm_panel_trace2,
        # yaxis='y10', line=dict(width=1),
        yaxis='y10', line=dict(width=1.2),  # darker line for trace2
        marker=dict(color=btm_trace2_color), hoverinfo='all',
        name=btm_panel_text2, showlegend=False,
    )
    data.append(btm_trace2)

    if plot_chart:
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=chart_path+chart_name)

    return cache



from myUtils import pickle_load  # NOQA

verbose = False  # True prints more output
# verbose = True  # True prints more output

path_dir = "C:/Users/ping/MyDrive/stocks/MktCap2b_AUMtop1200/"
path_data_dump = path_dir + "VSCode_dump/"
path_symbols_file = path_dir + "source/"
filename_symbols = path_symbols_file + '2021_Top1200_MktCap_n_AUM.txt'  # symbols text file
filename_pickled_df_OHLCV_all_dates = 'df_OHLCV_all_dates'  # pickled filename
filename_pickled_df_OHLCV = 'df_OHLCV'  # pickled filename reindexed to NYSE dates
filename_pickled_df_symbols_close = "df_symbols_close"  # pickled filename
filename_pickled_symbols_df_OHLCV =  'symbols_df_OHLCV'  # pickled filename


# retrieve pickled files
print(f"Full path to pickled df_OHLCV_all_dates:  {path_data_dump}{filename_pickled_df_OHLCV_all_dates}")
df_all_dates = pickle_load(path_data_dump, filename_pickled_df_OHLCV_all_dates, verbose=verbose)
print(f"Full path to pickled df_OHLCV:  {path_data_dump}{filename_pickled_df_OHLCV}")
df = pickle_load(path_data_dump, filename_pickled_df_OHLCV, verbose=verbose)
print(f"Full path to pickled symbols_df_OHLCV:  {path_data_dump}{filename_pickled_symbols_df_OHLCV}")
df_close = pickle_load(path_data_dump, filename_pickled_df_symbols_close, verbose=verbose)
print(f"Full path to pickled df_symbols_close:  {path_data_dump}{filename_pickled_df_symbols_close}")
symbols_df = pickle_load(path_data_dump, filename_pickled_symbols_df_OHLCV, verbose=verbose)

# from work import yf_plot_symbols  # NOQA
# symbols = ['FTEC', 'BCI', 'BTC-USD', 'ETH-USD']
symbols = ['FTEC', 'BCI']
# last date of data
date_end_limit = '2022-08-11'
# date_end_limit = dt.date.today().strftime("%Y-%m-%d")
iloc_offset = 252  # number of days to plot

# yf_plot_symbols
# symbols=symbols,
# path_data_dump=path_data_dump,
# filename_pickled_df_OHLCV = filename_pickled_df_OHLCV
date_start_limit=None
# date_end_limit=date_end_limit,
# iloc_offset=iloc_offset,



# from myUtils import pickle_load, dates_within_limits
# from yf_CStick_DD_OBV_UIMW_Diff_UIMW_cache import yf_candlestick



print('+'*15 + '  yf_plot_symbols(symbols, filename_pickled_df_OHLCV, path_data_dump)  ' + '+'*15 + '\n')
# directories and file names
df_filename = filename_pickled_df_OHLCV
# list of cache. Stores cache returned by candlestick plot
caches_returned_by_plot = []
# read df
df_OHLCV = pickle_load(path_data_dump, df_filename)
print(f"Full path to pickled df_OHLCV:  {path_data_dump}{df_filename}")
date_index_all_dates = df_OHLCV.index
# get iloc of date_end_limit in XOM's date index
date_start, date_end, iloc_date_start, iloc_date_end = \
    dates_within_limits(date_index_all_dates, date_start_limit,
                        date_end_limit, iloc_offset)
for symbol in symbols:
    # df = df_OHLCV[symbol]
    df = df_OHLCV[symbol][date_start:date_end]
    print('plot symbol {} from {} to {}\n'
            .format(symbol, date_start, date_end))
    cache_returned_by_plot = yf_candlestick(symbol, df, plot_chart=True)
    caches_returned_by_plot.append(cache_returned_by_plot)

