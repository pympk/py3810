def symb_perf_stats_vectorized_v8_(df_symbols_close):
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


