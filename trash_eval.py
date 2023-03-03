
def _7_perf_eval_v2(df_close):
    """
    df_close is a dataframe with date index, columns of symbols' closing price, and symbols as column names.
    The function first calculates symbols' drawdown, Ulcer-Index, max-drawdown, std(returns),
    std(returns)/Ulcer-Index, CAGR, CAGR/std(returns), CAGR/Ulcer-Index. Then it calculates and returns
    the group's (i.e. all the symbols) mean, standard-deviation, mean/standard-deviation for
    'return_std/UI', 'CAGR/return_std', 'CAGR/UI'.

    Args:
      df_close(dataframe): dataframe of symbols' close with
        DatetimeIndex e.g. (['2016-12-19', ... '2016-12-22']), symbols as
        column names, and symbols' close as column values.

    Return:
      df_perf(dataframe): dataframe with columns symbol, first date, last date, Year, CAGR,	UI,
          return_std/UI, CAGR/return_std, CAGR/UI
      grp_retnStd_d_UI(list): [(std(returns)/Ulcer-Index).mean, (std(returns)/Ulcer-Index).std,
        (std(returns)/Ulcer-Index).mean / (std(returns)/Ulcer-Index).std]
      grp_CAGR_d_retnStd(list): [(CAGR/std(returns)).mean, (CAGR/std(returns)).std,
        (CAGR/std(returns)).mean / (CAGR/std(returns)).std]
      grp_CAGR_d_UI(list): [(CAGR/Ulcer_Index).mean, (CAGR/Ulcer_Index).std,
        (CAGR/Ulcer_Index).mean / (CAGR/Ulcer_Index).std]
      grp_CAGR(list): [CAGR.mean, CAGR.std, CAGR.mean / CAGR.std]        
    """
    # v1 added grp_CAGR, ignore divide by zero

    import pandas as pd
    import numpy as np
    from myUtils import symb_perf_stats_vectorized_v8    

    (
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
    ) = symb_perf_stats_vectorized_v8(df_close)

    # caches_perf_stats_vect = []
    # for symbol in symbols:
    #     date_first = df_close.index[0].strftime("%Y-%m-%d")
    #     date_last = df_close.index[-1].strftime("%Y-%m-%d")

    #     cache = (
    #         symbol,
    #         date_first,
    #         date_last,
    #         period_yr,
    #         CAGR[symbol],
    #         UI[symbol],
    #         retnStd_div_UI[symbol],
    #         CAGR_div_retnStd[symbol],
    #         CAGR_div_UI[symbol],
    #     )
    #     # append performance data (tuple) to caches_perf_stats (list)
    #     caches_perf_stats_vect.append(cache)

    # column_names = [
    #     "symbol",
    #     "first date",
    #     "last date",
    #     "Year",
    #     "CAGR",
    #     "UI",
    #     "return_std/UI",
    #     "CAGR/return_std",
    #     "CAGR/UI",
    # ]
    # # write symbols' performance stats to dataframe
    # df_perf = pd.DataFrame(caches_perf_stats_vect, columns=column_names)

    # _cols = ["CAGR", "UI", "retrun_std/UI", "CAGR/return_std", "CAGR/UI"]




    # return df_perf, grp_retnStd_div_UI, grp_CAGR_div_retnStd, grp_CAGR_div_UI, grp_CAGR
    return grp_retnStd_div_UI, grp_CAGR_div_retnStd, grp_CAGR_div_UI, grp_CAGR

