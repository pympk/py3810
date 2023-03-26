def _2_split_train_val_test(
    df, s_train=0.7, s_val=0.2, s_test=0.1, verbose=False
):
    """Split df into training (df_train), validation (df_val)
    and test (df_test) and returns the splitted dfs

    Args:
        df(dataframe): dataframe to be splitted
        s_train(float): ratio of df_train / df, default = 0.7
        s_val(float): ratio of df_val / df, default = 0.2
        s_test(float): ratio of df_test / df, default = 0.1

    Return:
        df_train(dataframe): training portion of df
        df_val(dataframe): validation portion of df
        df_test(dataframe): test portion of df
    """

    if (s_train + s_val + s_test) - 1 > 0.0001:  # allow 0.01% error
        _sum = s_train + s_val + s_test
        raise Exception(
            f"s_train({s_train}) + s_val({s_val}) + s_test({s_test}) = \
                s_sum({_sum}), It must sums to 1"
        )
    n_train = round(len(df) * s_train)
    n_val = round(len(df) * s_val)
    # n_test = round(len(df) * s_test)
    df_train = df.iloc[0:n_train]
    df_val = df.iloc[n_train : (n_train + n_val)]
    df_test = df.iloc[(n_train + n_val) : :]

    len_df = len(df)
    len_train = len(df_train)
    len_val = len(df_val)
    len_test = len(df_test)
    sum_train_val_test = len_df + len_train + len_test

    if verbose:
        print(f"len(df): {len_df}")
        print(f"len(df_train): {len_train}")
        print(f"len(df_val): {len_val}")
        print(f"len(df_test): {len_test}")
        print(f"sum_train_val_test: {sum_train_val_test}")

    return df_train, df_val, df_test

def _3_random_slices(len_df, n_samples, days_lookback, days_eval, verbose=False):
    """Returns a list of random tuples of start_train, end_train, end_eval, where
    iloc[start_train:end_train] is used for training,
    and iloc[end_train:end_eval] is used for evaluation.  The length of the
    list is equal to n_samples.
    i.e. [(248, 368, 388), (199, 319, 339), ... (45, 165, 185)]

    Args:
        len_df(int): length of dataframe
        n_samples(int): number of slices to return
        days_lookback(int):  number of days to lookback for training
        days_eval(int): number of days forward for evaluation

    Return:
        r_slices(list of tuples): list of random tuples of start_train,
        end_train, end_eval, where iloc[start_train:end_train] is used for
        training, and iloc[end_train:end_eval] is used for evaluation.
          i.e. [(248, 368, 388), (199, 319, 339), ... (45, 165, 185)]
    """
    # v1 correct out-of-bound end_eval
    # v2 2023-03-11 check df generated from the random slices 

    # import random
    from random import randint

    # random.seed(0)
    n_sample = 0
    days_total = days_lookback + days_eval
    if verbose:
        print(
            f"days_lookback: {days_lookback}, days_eval: {days_eval}, days_total: {days_total}, len_df: {len_df}"
        )

    if days_total > len_df:    
        msg_err = f"days_total: {days_total} must be less or equal to len_df: {len_df}"
        raise SystemExit(msg_err)

    # random slices of iloc for train and eval that fits the constraints:
    #  days_lookback + days_eval > len_df
    #  start_train >= 0
    #  end_train - start_train = days_lookback
    #  end_eval - end_train = days_eval
    #  end_eval <= len_df
    r_slices = []
    while n_sample < n_samples:
        n_rand = randint(0, len_df)
        start_train = n_rand - days_lookback
        end_train = n_rand
        # start_eval = n_rand
        end_eval = n_rand + days_eval
        if 0 <= start_train and end_eval <= len_df:              
            r_slices.append((start_train, end_train, end_eval))
            n_sample += 1

    return r_slices

def _4_lookback_slices(max_slices, days_lookbacks, verbose=False):
    """
    Create sets of sub-slices from max_slices and days_lookbacks. A slice is
    a tuple of iloc values for start_train:end_train=start_eval:end_eval.
    Given 2 max_slices of [(104, 224, 234), (626, 746, 756)], it returns 2 sets
    [[(194, 224, 234), (164, 224, 234), (104, 224, 234)],
    [(716, 746, 756), (686, 746, 756), (626, 746, 756)]]. End_train is constant
    for each set. End_train-start_train is the same value from max_slice.

    Args:
        max_slices(list of tuples): list of iloc values for
        start_train:end_train=start_eval:end_eval, where end_train-start_train
        is the max value in days_lookbacks
        days_lookback(int):  number of days to lookback for training

    Return:
        lb_slices(list of lists of tuples): each sublist is set of iloc for
        start_train:end_train:end_eval tuples, where the days_lookbacks are
        the values of end_train-start_train in the set, and end_train-end_eval
        are same values from max_slices. The number of sublist is equal to
        number of max_slices. The number of tuples in the sublists is equal to
        number
    """

    lb_slices = []
    days_lookbacks.sort()  # sort list of integers in ascending order
    for max_slice in max_slices:
        l_max_slice = []
        for days in days_lookbacks:
            new_slice = (max_slice[1] - days, max_slice[1], max_slice[2])
            l_max_slice.append(new_slice)
            if verbose:
                print(f"days: {days}, {new_slice}")
        lb_slices.append(l_max_slice)
        if verbose:
            print("")

    return lb_slices

def _5_perf_ranks_old(df_close, days_lookbacks, verbose=False):
    """Returns perf_ranks_dict(dic. of dic. of symbols ranked in descending
     performance) and ranked_perf_ranks_dict(dic. of symbols ranked in
     descending frequency in a combined pool of symbols in perf_ranks_dict).

    Args:
        df_close(dataframe): dataframe of symbols' close with
         DatetimeIndex e.g. (['2016-12-19', ... '2016-12-22']), symbols as
         column names, and symbols' close as column values.
        days_lookbacks(list of positive integers): list of number of days to
        look-back, e.g. [15, 30], for performance calculation.

    Return:
        perf_ranks_dict({dic): dic. of dic. of symbols ranked in descending
         performance.
         First dic keys are:
          'period' + str(days_lookbacks[0]), ... ,
          'period' + str(days_lookbacks[-1])
         Second dic keys are:
          'r_CAGR/UI', 'r_CAGR/retnStd' and 'r_retnStd/UI'
         e.g.:
          {
            period-15': {
                         'r_CAGR/UI':  ['HZNP', ... , 'CB'],
                         'r_CAGR/retnStd': ['BBW', ... , 'CPRX'],
                         'r_retnStd/UI':   ['ENR', ... , 'HSY']
                        },
            ... ,
            'period-60': {
                          'r_CAGR/UI':  ['WNC', ... , 'FSLR'],
                          'r_CAGR/retnStd': ['VCYT', ... , 'BERY'],
                          'r_retnStd/UI':   ['MYOV', ... , 'NSC']
                         }
          }
        ranked_perf_ranks_dict(dic): dic. of symbols ranked in descending
         frequency in a combined pool of symbols in perf_ranks_dict.  Key is
         'ranked_perf_ranks_period' + str(days_lookbacks), e.g.:
         {'ranked_perf_ranks_period[-15, -30]': ['HZNP', ... , 'NSC']}
    """

    import pandas as pd
    from myUtils import symb_perf_stats_vectorized_v8

    perf_ranks_dict = {}  # dic of performance ranks
    syms_perf_rank = []  # list of lists to store top 100 ranked symbols

    for days_lookback in days_lookbacks:
        days_lookback = -1 * days_lookback
        f_name = "period" + str(days_lookback)
        _df_c = df_close[days_lookback::]
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
        ) = symb_perf_stats_vectorized_v8(_df_c)

        caches_perf_stats_vect = []
        for symbol in symbols:
            date_first = drawdown.index[0].strftime("%Y-%m-%d")
            date_last = drawdown.index[-1].strftime("%Y-%m-%d")
            cache = (
                symbol,
                date_first,
                date_last,
                period_yr,
                CAGR[symbol],
                UI[symbol],
                retnStd_div_UI[symbol],
                CAGR_div_retnStd[symbol],
                CAGR_div_UI[symbol],
            )
            # append performance data (tuple) to caches_perf_stats (list)
            caches_perf_stats_vect.append(cache)
        column_names = [
            "symbol",
            "first date",
            "last date",
            "Year",
            "CAGR",
            "UI",
            "retnStd/UI",
            "CAGR/retnStd",
            "CAGR/UI",
        ]

        # write symbols' performance stats to dataframe
        df_ps = pd.DataFrame(caches_perf_stats_vect, columns=column_names)
        df_ps["r_CAGR/UI"] = df_ps["CAGR/UI"].rank(ascending=False)
        df_ps["r_CAGR/retnStd"] = df_ps["CAGR/retnStd"].rank(ascending=False)
        df_ps["r_retnStd/UI"] = df_ps["retnStd/UI"].rank(ascending=False)

        _dict = {}
        cols_sort = ["r_CAGR/UI", "r_CAGR/retnStd", "r_retnStd/UI"]

        # print(f'{f_name} top 100 symbols')
        for col in cols_sort:
            symbols_top_n = (
                # df_ps.sort_values(by=[col]).head(n_symbols).symbol.values
                df_ps.sort_values(by=[col]).symbol.values
            )
            syms_perf_rank.append(list(symbols_top_n))
            # print(f'{col}: {symbols_top_n}')
            _dict[col] = symbols_top_n
            perf_ranks_dict[f"{f_name}"] = _dict

    syms_perf_rank  # list of lists of top 100 rank
    l_syms_perf_rank = [
        val for sublist in syms_perf_rank for val in sublist
    ]  # flatten list of lists

    from collections import Counter

    cnt_symbol_freq = Counter(l_syms_perf_rank)  # count symbols and frequency
    # print(cnt_symbol_freq)
    l_tuples = (
        cnt_symbol_freq.most_common()
    )  # convert to e.g [('AKRO', 6), ('IMVT', 4), ... ('ADEA', 3)]
    symbols_ranked_perf_ranks = [
        symbol for symbol, count in l_tuples
    ]  # select just the symbols without the frequency counts

    ranked_perf_ranks_dict = {}
    f_name = f"ranked_perf_ranks_period" + str(
        days_lookbacks
    )  # key name, ranked_perf_ranks_dict
    ranked_perf_ranks_dict[
        f"{f_name}"
        # values: list of most common symbols in all performance ranks in
        #  descending order
    ] = symbols_ranked_perf_ranks

    return perf_ranks_dict, ranked_perf_ranks_dict

def _5_perf_ranks(df_close, n_top_syms, verbose=False):
    """Returns perf_ranks(dic. of dic. of symbols ranked in descending
     performance) and most_common_syms(list of tuples of the most common
     symbols in perf_ranks in descending frequency).
     Example of perf_ranks:
        {'period-120': {'r_CAGR/UI': array(['CDNA', 'AVEO', 'DVAX', 'XOMA',
         'BLFS'], dtype=object),
        'r_CAGR/retnStd': array(['CDNA', 'AVEO', 'CYRX', 'BLFS', 'XOMA'],
         dtype=object),
        'r_retnStd/UI': array(['GDOT', 'DVAX', 'XOMA', 'CTRA', 'FTSM'],
         dtype=object)}}
     Example of most_common_syms:
        [('XOMA', 3), ('CDNA', 2), ('AVEO', 2), ('DVAX', 2), ('BLFS', 2),
         ('CYRX', 1), ('GDOT', 1), ('CTRA', 1), ('FTSM', 1)]

    Args:
        df_close(dataframe): dataframe of symbols' close with
         DatetimeIndex e.g. (['2016-12-19', ... '2016-12-22']), symbols as
         column names, and symbols' close as column values.
        n_top_syms(int): number of top symbols to keep in perf_ranks

    Return:
        perf_ranks({dic): dic. of dic. of symbols ranked in descending
         performance.
         First dic key is: 'period-' + str(days_lookback)
         Second dic keys are:
          'r_CAGR/UI', 'r_CAGR/retnStd' and 'r_retnStd/UI'
         e.g.:
          {
            period-15': {
                         'r_CAGR/UI':  ['HZNP', ... , 'CB'],
                         'r_CAGR/retnStd': ['BBW', ... , 'CPRX'],
                         'r_retnStd/UI':   ['ENR', ... , 'HSY']
                        },

        most_common_syms(list of tuples): list of tuples of symbols:frequency
         in descending frequency count of symbols in perf_ranks.  Key is
         'ranked_perf_ranks_period-' + str(days_lookbacks), e.g.:
         {'ranked_perf_ranks_period-15': ['HZNP', ... , 'NSC']}
    """

    import pandas as pd
    from collections import Counter
    from myUtils import symb_perf_stats_vectorized_v8    

    # dic of  dic of performance ranks
    # e.g. {'period-120': {'r_CAGR/UI': array(['LRN', 'APPS', 'FTSM', 'AU',
    #       'GRVY'], dtype=object),
    #      'r_CAGR/retnStd': array(['APPS', 'AU', 'LRN', 'GRVY', 'ERIE'],
    #       dtype=object),
    #      'r_retnStd/UI': array(['LRN', 'FTSM', 'AXSM', 'RCII', 'AU'],
    #       dtype=object)}}
    perf_ranks = {}
    syms_perf_rank = []  # list of lists to store top n ranked symbols

    days_lookback = len(df_close)
    f_name = "period-" + str(days_lookback)
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

    caches_perf_stats = []  # list of tuples in cache
    for symbol in symbols:
        # date_first = drawdown.index[0].strftime("%Y-%m-%d")
        # date_last = drawdown.index[-1].strftime("%Y-%m-%d")
        date_first = DD.index[0].strftime("%Y-%m-%d")  # drawdown
        date_last = DD.index[-1].strftime("%Y-%m-%d")        
        cache = (
            symbol,
            date_first,
            date_last,
            period_yr,
            CAGR[symbol],
            UI[symbol],
            retnStd_div_UI[symbol],
            CAGR_div_retnStd[symbol],
            CAGR_div_UI[symbol],
        )
        # append performance data (tuple) to caches_perf_stats (list)
        caches_perf_stats.append(cache)
    column_names = [
        "symbol",
        "first date",
        "last date",
        "Year",
        "CAGR",
        "UI",
        "retnStd/UI",
        "CAGR/retnStd",
        "CAGR/UI",
    ]

    # write symbols' performance stats to dataframe
    df_ps = pd.DataFrame(caches_perf_stats, columns=column_names)
    # create rank columns for performance stats
    df_ps["r_CAGR/UI"] = df_ps["CAGR/UI"].rank(ascending=False)
    df_ps["r_CAGR/retnStd"] = df_ps["CAGR/retnStd"].rank(ascending=False)
    df_ps["r_retnStd/UI"] = df_ps["retnStd/UI"].rank(ascending=False)

    _dict = {}
    cols_sort = ["r_CAGR/UI", "r_CAGR/retnStd", "r_retnStd/UI"]

    # print(f'{f_name} top 100 symbols')
    for col in cols_sort:
        symbols_top_n = (
            # sort df by column in col and return only the top symbols
            df_ps.sort_values(by=[col])
            .head(n_top_syms)
            .symbol.values
        )
        syms_perf_rank.append(list(symbols_top_n))
        _dict[col] = symbols_top_n
        perf_ranks[f"{f_name}"] = _dict

    syms_perf_rank  # list of lists of top n_top_syms symbols
    # flatten list of lists
    l_syms_perf_rank = [val for sublist in syms_perf_rank for val in sublist]

    cnt_symbol_freq = Counter(l_syms_perf_rank)  # count symbols and frequency
    # convert cnt_symbol_freq from counter obj. to list of tuples
    most_common_syms = (
        cnt_symbol_freq.most_common()
    )  # convert to e.g [('AKRO', 6), ('IMVT', 4), ... ('ADEA', 3)]

    return perf_ranks, most_common_syms

def _6_grp_tuples_sort_sum(l_tuples, reverse=True):
    # https://stackoverflow.com/questions/2249036/grouping-python-tuple-list
    # https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
    """
    Given a list of tuples of (key:value) such as:
    [('grape', 100), ('apple', 15), ('grape', 3), ('apple', 10),
     ('apple', 4), ('banana', 3)]
    Returns list of grouped-sorted-tuples based on summed-values such as:
    [('grape', 103), ('apple', 29), ('banana', 3)]

    Args:
        l_tuples(list of tuples): list of tuples of key(str):value(int) pairs
        reverse(bool): sort order of summed-values of the grouped tuples,
         default is descending order.

    Return:
        grp_sorted_list(list of tuples): list of grouped-sorted-tuples
         based on summed-values such as:
         [('grape', 103), ('apple', 29), ('banana', 3)]
    """

    import itertools
    from operator import itemgetter

    grp_list = []
    sorted_tuples = sorted(l_tuples)
    it = itertools.groupby(sorted_tuples, itemgetter(0))

    for key, subiter in it:
        # print(f'key: {key}')
        key_sum = sum(item[1] for item in subiter)
        # print(f'key_sum: {key_sum}')
        grp_list.append((key, key_sum))

    grp_sorted_list = sorted(grp_list, key=itemgetter(1), reverse=reverse)

    return grp_sorted_list

def _7_perf_eval_(df_close):
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
    """

    import pandas as pd
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

    caches_perf_stats_vect = []
    for symbol in symbols:
        date_first = df_close.index[0].strftime("%Y-%m-%d")
        date_last = df_close.index[-1].strftime("%Y-%m-%d")

        cache = (
            symbol,
            date_first,
            date_last,
            period_yr,
            CAGR[symbol],
            UI[symbol],
            retnStd_div_UI[symbol],
            CAGR_div_retnStd[symbol],
            CAGR_div_UI[symbol],
        )
        # append performance data (tuple) to caches_perf_stats (list)
        caches_perf_stats_vect.append(cache)

    column_names = [
        "symbol",
        "first date",
        "last date",
        "Year",
        "CAGR",
        "UI",
        "return_std/UI",
        "CAGR/return_std",
        "CAGR/UI",
    ]
    # write symbols' performance stats to dataframe
    df_perf = pd.DataFrame(caches_perf_stats_vect, columns=column_names)

    _cols = ["CAGR", "UI", "retrun_std/UI", "CAGR/return_std", "CAGR/UI"]

    _mean = df_perf["CAGR/UI"].mean()
    _std = df_perf["CAGR/UI"].std()
    grp_CAGR_d_UI = [_mean, _std, _mean / _std]

    _mean = df_perf["CAGR/return_std"].mean()
    _std = df_perf["CAGR/return_std"].std()
    grp_CAGR_d_retnStd = [_mean, _std, _mean / _std]

    _mean = df_perf["return_std/UI"].mean()
    _std = df_perf["return_std/UI"].std()
    grp_retnStd_d_UI = [_mean, _std, _mean / _std]

    return df_perf, grp_retnStd_d_UI, grp_CAGR_d_retnStd, grp_CAGR_d_UI

def _7_perf_eval_v1(df_close):
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

    caches_perf_stats_vect = []
    for symbol in symbols:
        date_first = df_close.index[0].strftime("%Y-%m-%d")
        date_last = df_close.index[-1].strftime("%Y-%m-%d")

        cache = (
            symbol,
            date_first,
            date_last,
            period_yr,
            CAGR[symbol],
            UI[symbol],
            retnStd_div_UI[symbol],
            CAGR_div_retnStd[symbol],
            CAGR_div_UI[symbol],
        )
        # append performance data (tuple) to caches_perf_stats (list)
        caches_perf_stats_vect.append(cache)

    column_names = [
        "symbol",
        "first date",
        "last date",
        "Year",
        "CAGR",
        "UI",
        "return_std/UI",
        "CAGR/return_std",
        "CAGR/UI",
    ]
    # write symbols' performance stats to dataframe
    df_perf = pd.DataFrame(caches_perf_stats_vect, columns=column_names)

    _cols = ["CAGR", "UI", "retrun_std/UI", "CAGR/return_std", "CAGR/UI"]


    # ignore divide by zero in case of just one symbol (i.e. SPY) where std is 0, mean/std is inf
    with np.errstate(divide='ignore'):
        _mean = df_perf["CAGR/UI"].mean()
        _std = df_perf["CAGR/UI"].std(ddof=0)
        grp_CAGR_d_UI = [_mean, _std, _mean / _std]

        _mean = df_perf["CAGR/return_std"].mean()
        _std = df_perf["CAGR/return_std"].std(ddof=0)
        grp_CAGR_d_retnStd = [_mean, _std, _mean / _std]

        _mean = df_perf["return_std/UI"].mean()
        _std = df_perf["return_std/UI"].std(ddof=0)
        grp_retnStd_d_UI = [_mean, _std, _mean / _std]

        grp_CAGR = [grp_MeanCAGR, grp_StdCAGR, grp_MeanCAGR/grp_StdCAGR]    

    return df_perf, grp_retnStd_d_UI, grp_CAGR_d_retnStd, grp_CAGR_d_UI, grp_CAGR

def top_set_sym_freq_cnt(top_set_syms_n_freq):
    """
    Top_set_syms_n_freq is a list of tuples. Each tuple is a pair of symbol:frequency_count, i.e.: 
      [('LNTH', 6), ('TNK', 6), ('FTSM', 5), ('EME', 4), ('FCN', 4), ('PRG', 3), ('SGEN', 3)]

    The function can accommodate upto 5 periods of days_lookbacks, i.e.:
      days_lookbacks = [5, 10, 15, 20, 25], len(days_lookbacks)*3 = 15)

    Returns a list of 14 lists. list[0], list[1], ..., list[14] are lists of symbols with
    frequency_count of 15, 14, ..., 2 respectively.

    Args:
      top_set_syms_n_freq(list): List of tuples. Each tuple is a pair of symbol:frequency_count, i.e.: 
        [('LNTH', 6), ('TNK', 6), ('FTSM', 5), ('EME', 4), ('FCN', 4), ('PRG', 3), ('SGEN', 3)]

    Return:
      l_sym_freq_cnt(list): list of 14 lists. list[0], list[1], ..., list[14] are lists of symbols with
          frequency_count of 15, 14, ..., 2 respectively.
    """

    sym_freq_cnt_15 = []
    sym_freq_cnt_14 = []
    sym_freq_cnt_13 = []
    sym_freq_cnt_12 = []
    sym_freq_cnt_11 = []
    sym_freq_cnt_10 = []
    sym_freq_cnt_9 = []
    sym_freq_cnt_8 = []
    sym_freq_cnt_7 = []
    sym_freq_cnt_6 = []
    sym_freq_cnt_5 = []
    sym_freq_cnt_4 = []
    sym_freq_cnt_3 = []
    sym_freq_cnt_2 = []

    for sym_n_freq in top_set_syms_n_freq:
        _sym, _freq = sym_n_freq[0], sym_n_freq[1]
        if _freq == 15:
            sym_freq_cnt_15.append(_sym)
        elif _freq == 14:
            sym_freq_cnt_14.append(_sym)
        elif _freq == 13:
            sym_freq_cnt_13.append(_sym)
        elif _freq == 12:
            sym_freq_cnt_12.append(_sym)                        
        elif _freq == 11:
            sym_freq_cnt_11.append(_sym)
        elif _freq == 10:
            sym_freq_cnt_10.append(_sym)            
        elif _freq == 9:
            sym_freq_cnt_9.append(_sym)
        elif _freq == 8:
            sym_freq_cnt_8.append(_sym)
        elif _freq == 7:
            sym_freq_cnt_7.append(_sym)  
        elif _freq == 6:
            sym_freq_cnt_6.append(_sym)
        elif _freq == 5:
            sym_freq_cnt_5.append(_sym)
        elif _freq == 4:
            sym_freq_cnt_4.append(_sym)
        elif _freq == 3:
            sym_freq_cnt_3.append(_sym)          
        else:
            sym_freq_cnt_2.append(_sym)

    l_sym_freq_cnt = []

    l_sym_freq_cnt.append(sym_freq_cnt_15)
    l_sym_freq_cnt.append(sym_freq_cnt_14)
    l_sym_freq_cnt.append(sym_freq_cnt_13)
    l_sym_freq_cnt.append(sym_freq_cnt_12)    
    l_sym_freq_cnt.append(sym_freq_cnt_11)   
    l_sym_freq_cnt.append(sym_freq_cnt_10)
    l_sym_freq_cnt.append(sym_freq_cnt_9)
    l_sym_freq_cnt.append(sym_freq_cnt_8)
    l_sym_freq_cnt.append(sym_freq_cnt_7)    
    l_sym_freq_cnt.append(sym_freq_cnt_6)
    l_sym_freq_cnt.append(sym_freq_cnt_5)
    l_sym_freq_cnt.append(sym_freq_cnt_4)
    l_sym_freq_cnt.append(sym_freq_cnt_3)    
    l_sym_freq_cnt.append(sym_freq_cnt_2)    

    return l_sym_freq_cnt
