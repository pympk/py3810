def split_train_val_test(
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

def random_slices(len_df, n_samples, days_lookback, days_eval, verbose=False):
    """
    Returns a list of random tuples of start_train, end_train, end_eval,
    where iloc[start_train:end_train] is used for training,
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
    # v3 move error check "days_total > len_df" to calling script 

    # import random
    from random import randint

    # initialize counter
    n_sample = 0  

    if verbose:
        print(
            f"days_lookback: {days_lookback}, days_eval: {days_eval}, days_total: {days_total}, len_df: {len_df}"
        )

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

def lookback_slices(max_slices, days_lookbacks, verbose=False):
    """
    Create sets of sub-slices from max_slices and days_lookbacks. A slice is
    a tuple of iloc values for start_train:end_train=start_eval:end_eval.
    Given 2 max_slices of [(104, 224, 234), (626, 746, 756)], it returns 2 sets
    [[(194, 224, 234), (164, 224, 234), (104, 224, 234)],
    [(716, 746, 756), (686, 746, 756), (626, 746, 756)]]. End_train is constant
    for each set. End_train - start_train is the value of the maximum slice.    


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

def perf_ranks_old(df_close, days_lookbacks, verbose=False):
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



def rank_perf(df_close, n_top_syms, verbose=False):
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



def grp_tuples_sort_sum(l_tuples, reverse=True):
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

def perf_eval_(df_close):
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

def perf_eval_v1(df_close):
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

def compare_picks_v_SPY(df_eval, df_SPY):
    import numpy as np
    from myUtils import symb_perf_stats_vectorized_v8 as sps

    _d_cols_values = {}

    if np.all(df_eval.index == df_SPY.index):
        (
            _symbols,
            _period_yr,
            _retn,
            _DD,
            _UI,
            _MDD,
            _retnMean,
            _retnStd,
            _retnStd_div_UI,
            _CAGR,
            _CAGR_div_retnStd,
            _CAGR_div_UI,
            SPY_retnStd_d_UI,
            SPY_CAGR,
            SPY_CAGR_d_retnStd,
            SPY_CAGR_d_UI,
        ) = sps(df_SPY)

        (
            _symbols,
            _period_yr,
            _retn,
            _DD,
            _UI,
            _MDD,
            _retnMean,
            _retnStd,
            _retnStd_div_UI,
            _CAGR,
            _CAGR_div_retnStd,
            _CAGR_div_UI,
            grp_retnStd_d_UI,
            grp_CAGR,
            grp_CAGR_d_retnStd,
            grp_CAGR_d_UI,
        ) = sps(df_eval)

        ### column names for rows ###
        col_add1 = ["grp(CAGR)_mean", "grp(CAGR)_std", "grp(CAGR)_mean/std"]
        col_add2 = ["grp(CAGR/UI)_mean", "grp(CAGR/UI)_std", "grp(CAGR/UI)_mean/std"]
        col_add3 = [
            "grp(CAGR/retnStd)_mean",
            "grp(CAGR/retnStd)_std",
            "grp(CAGR/retnStd)_mean/std",
        ]
        col_add4 = [
            "grp(retnStd/UI)_mean",
            "grp(retnStd/UI)_std",
            "grp(retnStd/UI)_mean/std",
        ]
        col_add5 = ["SPY_CAGR", "SPY_CAGR/UI", "SPY_CAGR/retnStd", "SPY_retnStd/UI"]
        cols = col_add1 + col_add2 + col_add3 + col_add4 + col_add5

        row_add1 = [grp_CAGR[0], grp_CAGR[1], grp_CAGR[2]]
        row_add2 = [grp_CAGR_d_UI[0], grp_CAGR_d_UI[1], grp_CAGR_d_UI[2]]
        row_add3 = [grp_CAGR_d_retnStd[0], grp_CAGR_d_retnStd[1], grp_CAGR_d_retnStd[2]]
        row_add4 = [grp_retnStd_d_UI[0], grp_retnStd_d_UI[1], grp_retnStd_d_UI[2]]
        row_add5 = [
            SPY_CAGR[0],
            SPY_CAGR_d_UI[0],
            SPY_CAGR_d_retnStd[0],
            SPY_retnStd_d_UI[0],
        ]
        rows = row_add1 + row_add2 + row_add3 + row_add4 + row_add5

        _d_cols_values = dict(zip(cols, rows))

    return _d_cols_values

def _get_grp_top_syms_n_freq(df, list_random_lookback_slices, days_lookbacks, n_top_syms, syms_start, syms_end, verbose=False):
    from yf_utils import rank_perf, grp_tuples_sort_sum

    # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
    #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
    #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
    #   where each list is the best performing symbols from a lb_slices, e.g.
    #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
    grp_top_set_syms_n_freq = (
        []
    )  # list of lists of top_set_symbols_n_freq, there are n_samples lists in list
    grp_top_set_syms = []  # grp_top_set_syms_n_freq without the frequency count

    dates_end_df_train = []  # store [date_end_df_train, ...]

    # lb_slices, e.g  [(483, 513, 523), (453, 513, 523), (393, 513, 523)],
    #  is one max_lookback_slice, e.g. (393, 513, 523), along with
    #  the remaining slices of the days_lookbacks, e.g. (483, 513, 523), (453, 513, 523)
    for i, lb_slices in enumerate(list_random_lookback_slices):
        print(
            f"\n########## {i + 1} of {len(list_random_lookback_slices)} lb_slices: {lb_slices} in list_random_lookback_slices ##########"
        )

        # unsorted list of the most frequent symbols in performance metrics of the lb_slices
        grp_most_freq_syms = []
        for j, lb_slice in enumerate(lb_slices):  # lb_slice, e.g. (246, 276, 286)
            iloc_start_train = lb_slice[0]  # iloc of start of training period
            iloc_end_train = lb_slice[1]  # iloc of end of training period
            iloc_start_eval = iloc_end_train  # iloc of start of evaluation period
            iloc_end_eval = lb_slice[2]  # iloc of end of evaluation period
            lookback = iloc_end_train - iloc_start_train
            d_eval = iloc_end_eval - iloc_start_eval

            _df = df.iloc[iloc_start_train:iloc_end_train]
            date_start_df_train = _df.index[0].strftime("%Y-%m-%d")
            date_end_df_train = _df.index[-1].strftime("%Y-%m-%d")

            if verbose:
                print(
                    f"days lookback:       {lookback},  {j + 1} of {len(days_lookbacks)} days_lookbacks: {days_lookbacks}"
                )
                print(f"lb_slices:           {lb_slices}")
                print(f"lb_slice:            {lb_slice}")
                print(f"days eval:           {d_eval}")
                print(f"iloc_start_train:    {iloc_start_train}")
                print(f"iloc_end_train:      {iloc_end_train}")
                print(f"date_start_df_train: {date_start_df_train}")
                print(f"date_end_df_train:   {date_end_df_train}")

            # perf_ranks, performance rank of one day in days_loolbacks (30 of [15, 30, 60]) e.g.:
            # {'period-30': {'r_CAGR/UI': array(['BMA', 'ACIW', 'SHV', 'GBTC', 'BVH', ..., 'BGNE', 'HLF', 'WB', 'HZNP']
            #                'r_CAGR/retnStd': array(['BMA', 'GBTC', 'EDU', 'PAR', ..., 'BGNE', 'ACIW', 'HLF', 'EXAS']
            #                'r_retnStd/UI': array(['HZNP', 'SHV', 'BVH', 'ACIW', ..., 'FTSM', 'COLL', 'ALGN', 'JOE']
            # most_freq_syms, most frequency symbols of one day in days_loolbacks (30 of [15, 30, 60]) e.g.:
            # [('BMA', 3), ('ACIW', 3), ('BVH', 3), ('BGNE', 3), ('SHV', 2), ..., ('GBTC', 2), ('AJRD', 1), ('TGH', 1)]
            perf_ranks, most_freq_syms = rank_perf(_df, n_top_syms=n_top_syms)
            # unsorted accumulation of most_freq_syms for all the days in days_lookbacks
            grp_most_freq_syms.append(most_freq_syms)
            if verbose:
                # one lookback of r_CAGR/UI, r_CAGR/retnStd, r_retnStd/UI, e.g. 30 of days_lookbacks [15, 30, 60]
                print(f"perf_ranks: {perf_ranks}")
                # most common symbols of perf_ranks
                print(f"most_freq_syms: {most_freq_syms}")
                # grp_perf_ranks[lookback] = perf_ranks
                print(f"+++ finish lookback slice {lookback} +++\n")

        if verbose:
            print(f"grp_most_freq_syms: {grp_most_freq_syms}")
            # grp_most_freq_syms a is list of lists of tuples of
            #  the most-common-symbols symbol:frequency cumulated from
            #  each days_lookback
            print(f"**** finish lookback slices {lb_slices} ****\n")

        # flatten list of lists of (symbol:frequency_count) pairs of all the days in days_lookbacks
        flat_grp_most_freq_syms = [
            val for sublist in grp_most_freq_syms for val in sublist
        ]
        # sum symbol's frequency_count, return sorted tuples of "symbol:frequency_count"
        #  in descending order for all the days in days_lookbacks
        #  e.g. [('TA', 6), ('OEC', 4), ('SHV', 4), ('HY', 3), ('ELF', 2), ... , ('TNP', 1)]
        set_most_freq_syms = grp_tuples_sort_sum(
            flat_grp_most_freq_syms, reverse=True
        )
        # get the top n_top_syms of the most frequent "symbol, frequency" pairs
        top_set_syms_n_freq = set_most_freq_syms[0:n_top_syms]
        # get symbols from top_set_syms_n_freq, i[0] = symbol, i[1]=symbol's frequency count
        top_set_syms = [i[0] for i in top_set_syms_n_freq[syms_start:syms_end]]

        # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
        #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
        #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
        #   where each list is the best performing symbols from a lb_slices, e.g.
        #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
        grp_top_set_syms_n_freq.append(top_set_syms_n_freq)
        grp_top_set_syms.append(top_set_syms)
        dates_end_df_train.append(
            date_end_df_train
        )  # df_train end date for set of lookback slices

        if verbose:
            print(
                f"top {n_top_syms} ranked symbols and frequency from set {lb_slices}:\n{top_set_syms_n_freq}"
            )
            print(
                f"top {n_top_syms} ranked symbols from set {lb_slices}:\n{top_set_syms}"
            )
            print(
                f"===== finish top {n_top_syms} ranked symbols from days_lookback set {lb_slices} =====\n\n"
            )

    # return grp_top_set_syms_n_freq, grp_top_set_syms
    return grp_top_set_syms_n_freq, grp_top_set_syms, dates_end_df_train

def get_grp_top_syms_n_freq(df, list_random_lookback_slices, days_lookbacks, n_top_syms, syms_start, syms_end, verbose=False):
    from yf_utils import rank_perf, grp_tuples_sort_sum

    # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
    #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
    #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
    #   where each list is the best performing symbols from a lb_slices, e.g.
    #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
    grp_top_set_syms_n_freq = (
        []
    )  # list of lists of top_set_symbols_n_freq, there are n_samples lists in list
    grp_top_set_syms = []  # grp_top_set_syms_n_freq without the frequency count

    dates_end_df_train = []  # store [date_end_df_train, ...]

    # lb_slices, e.g  [(483, 513, 523), (453, 513, 523), (393, 513, 523)],
    #  is one max_lookback_slice, e.g. (393, 513, 523), along with
    #  the remaining slices of the days_lookbacks, e.g. (483, 513, 523), (453, 513, 523)
    for i, lb_slices in enumerate(list_random_lookback_slices):
        print(
            f"\n########## {i + 1} of {len(list_random_lookback_slices)} lb_slices: {lb_slices} in list_random_lookback_slices ##########"
        )

        # unsorted list of the most frequent symbols in performance metrics of the lb_slices
        grp_most_freq_syms = []
        for j, lb_slice in enumerate(lb_slices):  # lb_slice, e.g. (246, 276, 286)
            iloc_start_train = lb_slice[0]  # iloc of start of training period
            iloc_end_train = lb_slice[1]  # iloc of end of training period
            iloc_start_eval = iloc_end_train  # iloc of start of evaluation period
            iloc_end_eval = lb_slice[2]  # iloc of end of evaluation period
            lookback = iloc_end_train - iloc_start_train
            d_eval = iloc_end_eval - iloc_start_eval

            _df = df.iloc[iloc_start_train:iloc_end_train]
            date_start_df_train = _df.index[0].strftime("%Y-%m-%d")
            date_end_df_train = _df.index[-1].strftime("%Y-%m-%d")

            if verbose:
                print(
                    f"days lookback:       {lookback},  {j + 1} of {len(days_lookbacks)} days_lookbacks: {days_lookbacks}"
                )
                print(f"lb_slices:           {lb_slices}")
                print(f"lb_slice:            {lb_slice}")
                print(f"days eval:           {d_eval}")
                print(f"iloc_start_train:    {iloc_start_train}")
                print(f"iloc_end_train:      {iloc_end_train}")
                print(f"date_start_df_train: {date_start_df_train}")
                print(f"date_end_df_train:   {date_end_df_train}")

            # perf_ranks, performance rank of one day in days_loolbacks (30 of [15, 30, 60]) e.g.:
            # {'period-30': {'r_CAGR/UI': array(['BMA', 'ACIW', 'SHV', 'GBTC', 'BVH', ..., 'BGNE', 'HLF', 'WB', 'HZNP']
            #                'r_CAGR/retnStd': array(['BMA', 'GBTC', 'EDU', 'PAR', ..., 'BGNE', 'ACIW', 'HLF', 'EXAS']
            #                'r_retnStd/UI': array(['HZNP', 'SHV', 'BVH', 'ACIW', ..., 'FTSM', 'COLL', 'ALGN', 'JOE']
            # most_freq_syms, most frequency symbols of one day in days_loolbacks (30 of [15, 30, 60]) e.g.:
            # [('BMA', 3), ('ACIW', 3), ('BVH', 3), ('BGNE', 3), ('SHV', 2), ..., ('GBTC', 2), ('AJRD', 1), ('TGH', 1)]
            perf_ranks, most_freq_syms = rank_perf(_df, n_top_syms=n_top_syms)
            # unsorted accumulation of most_freq_syms for all the days in days_lookbacks
            grp_most_freq_syms.append(most_freq_syms)
            if verbose:
                # one lookback of r_CAGR/UI, r_CAGR/retnStd, r_retnStd/UI, e.g. 30 of days_lookbacks [15, 30, 60]
                print(f"perf_ranks: {perf_ranks}")
                # most common symbols of perf_ranks
                print(f"most_freq_syms: {most_freq_syms}")
                # grp_perf_ranks[lookback] = perf_ranks
                print(f"+++ finish lookback slice {lookback} +++\n")

        if verbose:
            print(f"grp_most_freq_syms: {grp_most_freq_syms}")
            # grp_most_freq_syms a is list of lists of tuples of
            #  the most-common-symbols symbol:frequency cumulated from
            #  each days_lookback
            print(f"**** finish lookback slices {lb_slices} ****\n")

        # flatten list of lists of (symbol:frequency_count) pairs of all the days in days_lookbacks
        flat_grp_most_freq_syms = [
            val for sublist in grp_most_freq_syms for val in sublist
        ]
        # sum symbol's frequency_count, return sorted tuples of "symbol:frequency_count"
        #  in descending order for all the days in days_lookbacks
        #  e.g. [('TA', 6), ('OEC', 4), ('SHV', 4), ('HY', 3), ('ELF', 2), ... , ('TNP', 1)]
        set_most_freq_syms = grp_tuples_sort_sum(
            flat_grp_most_freq_syms, reverse=True
        )
        # get the top n_top_syms of the most frequent "symbol, frequency" pairs
        top_set_syms_n_freq = set_most_freq_syms[0:n_top_syms]
        # get symbols from top_set_syms_n_freq, i[0] = symbol, i[1]=symbol's frequency count
        top_set_syms = [i[0] for i in top_set_syms_n_freq[syms_start:syms_end]]

        # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
        #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
        #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
        #   where each list is the best performing symbols from a lb_slices, e.g.
        #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
        grp_top_set_syms_n_freq.append(top_set_syms_n_freq)
        grp_top_set_syms.append(top_set_syms)
        dates_end_df_train.append(
            date_end_df_train
        )  # df_train end date for set of lookback slices

        if verbose:
            print(
                f"top {n_top_syms} ranked symbols and frequency from set {lb_slices}:\n{top_set_syms_n_freq}"
            )
            print(
                f"top {syms_start} to {syms_end} symbols from the top {n_top_syms} ranked symbols from set {lb_slices}:\n{top_set_syms}"
            )            
            print(
                f"===== finish top {n_top_syms} ranked symbols from days_lookback set {lb_slices} =====\n\n"
            )

    # return grp_top_set_syms_n_freq, grp_top_set_syms
    return grp_top_set_syms_n_freq, grp_top_set_syms, dates_end_df_train


def eval_grp_top_syms_n_freq(df, max_lookback_slices, sets_lookback_slices, grp_top_set_syms_n_freq, days_lookbacks, days_eval, n_samples,  n_top_syms, syms_start, syms_end, verbose):
    import pandas as pd
    from myUtils import symb_perf_stats_vectorized_v8

    print("z_grp_top_set_syms_n_freq:")
    # zip max_lookback_slices with top performing symbols:freq pairs
    z_grp_top_set_syms_n_freq = zip(max_lookback_slices, grp_top_set_syms_n_freq)

    idx_train = pd.Index([])
    idx_eval = pd.Index([])

    l_row_add_total = []
    print(f"type(l_row_add_total): {type(l_row_add_total)}")
    print(f"l_row_add_total: {l_row_add_total}\n\n")

    for i, (_lookback_slice, _top_set_syms_n_freq) in enumerate(
        z_grp_top_set_syms_n_freq
    ):
        iloc_start_train = _lookback_slice[0]
        iloc_end_train = _lookback_slice[1]
        iloc_start_eval = iloc_end_train
        iloc_end_eval = _lookback_slice[2]

        print(f"{i + 1 } of {n_samples} max_lookback_slice: {_lookback_slice}\n")
        # print(f'max_lookback_slice: {_lookback_slice}\n')
        # dates correspond to max_lookback_slice
        idx_train = df.index[iloc_start_train:iloc_end_train]
        date_start_df_train = idx_train[0].strftime("%Y-%m-%d")
        date_end_df_train = idx_train[-1].strftime("%Y-%m-%d")
        print(f"len(idx_train): {len(idx_train)}")
        print(f"idx_train: {idx_train}")
        print(
            f"df_train_dates (inclusive): {date_start_df_train} - {date_end_df_train}\n"
        )

        idx_eval = df.index[iloc_start_eval:iloc_end_eval]
        date_start_df_eval = idx_eval[0].strftime("%Y-%m-%d")
        date_end_df_eval = idx_eval[-1].strftime("%Y-%m-%d")
        print(f"len(idx_eval): {len(idx_eval)}")
        print(f"idx_eval: {idx_eval}")
        print(f"df_eval dates (inclusive): {date_start_df_eval} - {date_end_df_eval}\n")

        print(f"top_set_syms_n_freq: {_top_set_syms_n_freq}\n")

        l_sym_freq_cnt = top_set_sym_freq_cnt(_top_set_syms_n_freq)
        if verbose:
            print(f"set_lookback_slices: {sets_lookback_slices[i]}")
            print(f"max_lookback_slice:  {_lookback_slice}")
            print(f"days_lookbacks:      {days_lookbacks}")
            print(f"date_end_df_train:   {date_end_df_train}")
            print(f"sym_freq_15:         {l_sym_freq_cnt[0]}")
            print(f"sym_freq_14:         {l_sym_freq_cnt[1]}")
            print(f"sym_freq_13:         {l_sym_freq_cnt[2]}")
            print(f"sym_freq_12:         {l_sym_freq_cnt[4]}")
            print(f"sym_freq_10:         {l_sym_freq_cnt[5]}")
            print(f"sym_freq_9:          {l_sym_freq_cnt[6]}")
            print(f"sym_freq_8:          {l_sym_freq_cnt[7]}")
            print(f"sym_freq_7:          {l_sym_freq_cnt[8]}")
            print(f"sym_freq_6:          {l_sym_freq_cnt[9]}")
            print(f"sym_freq_5:          {l_sym_freq_cnt[10]}")
            print(f"sym_freq_4:          {l_sym_freq_cnt[11]}")
            print(f"sym_freq_3:          {l_sym_freq_cnt[12]}")
            print(f"sym_freq_2:          {l_sym_freq_cnt[13]}\n")

        if idx_eval.empty:
            msg_stop = (
                f'\n\n**** STOPPED df_eval is empty, the run_type is "{run_type}" ****'
            )
            raise SystemExit(msg_stop)

        _sym_idx = ["SPY"]
        df_SPY = df[iloc_start_eval:iloc_end_eval][_sym_idx]

        # drop list with frequency count 2 or less, in l_sym_freq_cnt from zip
        zip_cnt_n_syms = zip(
            [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3], l_sym_freq_cnt[:-1]
        )
        for item in zip_cnt_n_syms:
            sym_freq_cnt = item[0]
            syms = item[1]
            if syms:  # iterate ONLY if there are symbols in syms
                df_eval = df[iloc_start_eval:iloc_end_eval][syms]

                if verbose:
                    print(
                        f"iloc_start_eval: {iloc_start_eval:>6},  df_eval first date: {date_start_df_eval:>10}"
                    )
                    print(
                        f"iloc_end_eval:   {iloc_end_eval:>6},  df_eval last date:  {date_end_df_eval:>10}"
                    )
                    print(f"frequency count of symbol(s): {sym_freq_cnt}")

                    syms_n_SPY = syms + ["SPY"]
                    df_eval_n_SPY = df[iloc_start_eval:iloc_end_eval][syms_n_SPY]
                    print(f"\ndf_eval_n_SPY:\n{df_eval_n_SPY}\n")

                d_cols_values = compare_picks_v_SPY(df_eval, df_SPY)
                val_rows = []

                print(f"Function compare_picks_v_SPY returned column_name:value pairs")
                print(f'{"column_name":<45}{"value":<14}')
                for key_col in d_cols_values:
                    print(f"{key_col:<30}{d_cols_values[key_col]:>20,.4f}")
                    val_rows.append(d_cols_values[key_col])
                print("")

                row_add0 = [
                    n_samples,
                    str(days_lookbacks),
                    days_eval,
                    n_top_syms,
                    syms_start,
                    syms_end,
                    sym_freq_cnt,
                ]
                row_add_total = row_add0 + val_rows

                print(f"type(l_row_add_total): {type(l_row_add_total)}")
                print(f"l_row_add_total: {l_row_add_total}")
                print(f"type(row_add_total): {type(row_add_total)}")
                print(f"row_add_total: {row_add_total}")

                l_row_add_total.append(row_add_total)

        print("=" * 80, "\n\n\n")

    print(f"type(l_row_add_total): {type(l_row_add_total)}")
    print(f"l_row_add_total: {l_row_add_total}\n\n")

    return l_row_add_total


def best_perf_syms_sets_lookback_slices(
    df_close,
    sets_lookback_slices,
    n_top_syms=20,
    syms_start=0,
    syms_end=10,
    verbose=False,
):
    """
    Given df_close, a dataframe of symbols' close, and sets_lookback_slices,
    the script returns the best performing symbols for each lookback_slices.

    For a days_lookbacks=[30, 60, 120] and n_samples=2,
    a sets_lookback_slices might be:
      [[(403, 433, 433), (373, 433, 433), (313, 433, 433)],
       [(243, 273, 273), (213, 273, 273), (153, 273, 273)]]

    For the above sets_lookback_slices, the script returns 2 sets
    of best performing symbols. One for lookback_slices:
      [(403, 433, 433), (373, 433, 433), (313, 433, 433)]
    Another set of best performing symbols for lookback_slices:
      [(243, 273, 273), (213, 273, 273), (153, 273, 273)]

     Args:
      df_close(dataframe): dataframe of symbols' close with
        DatetimeIndex e.g. (['2016-12-19', ... '2016-12-22']), symbols as
        column names, and symbols' close as column values.
      sets_lookback_slices(list): a list of lists of tuples. The tuples are
        iloc values for (start_train:end_train=start_eval:end_eval). If
        n_sample=2 and days_lookbacks=[30, 60, 120], then an example of
        a sets_lookback_slices:
        [[(241, 271, 271), (211, 271, 271), (151, 271, 271)],
         [(126, 156, 156), (96, 156, 156), (36, 156, 156)]]
        where: The length of the list is equal to n_sample. The number
        of tuples in the sub-list is equal to the days in days_lookbacks.
        The tuple has the iloc values (start_train:end_train=start_eval:end_eval)
      n_top_syms(int): number of the most-common symbols from days_lookbacks'
        performance rankings to keep, default = 20
      syms_start(int): slice starts for selecting the best performing symbols,
        default=0
      syms_end(int): slice end for selecting the best performing symbols,
       default=10

    Return:
      grp_top_set_syms_n_freq(list): is a list of lists of the top n_top_syms of the
        best performing symbols, in terms of return_std/UI, CAGR/return_std, CAGR/UI,
        and their number of occurrence for sets_lookback_slices.
        The list of lists corresponds to days_lookbacks in l_sorted_days_lookbacks.
        e.g. grp_top_set_syms_n_freq:
          [[('GPS', 8), ('SHV', 8), ('FTSM', 7), ('GBTC', 7), ('BTC-USD', 6), ('CBOE', 6),
            ('ANF', 5), ('NRG', 5), ('WING', 5), ('DELL', 4), ('EDU', 4), ('HIBB', 4),
            ('LRN', 4), ('ALL', 3), ('CAH', 3), ('CMG', 3), ('GDDY', 3), ('HRB', 3),
            ('MDLZ', 3), ('PGR', 3)]]
      grp_top_set_syms(list): It is grp_top_set_syms_n_freq[syms_start:syms_end]
        with number of occurrence dropped.
        e.g. [['GPS', 'SHV', 'FTSM', 'GBTC', 'BTC-USD', 'CBOE', 'ANF', 'NRG', 'WING', 'DELL']]
      date_end_df_train(str): training df's last date, e.g. '2023-11-30'
    """

    from yf_utils import rank_perf, grp_tuples_sort_sum

    # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
    #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
    #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
    #   where each list is the best performing symbols from a lb_slices, e.g.
    #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
    # there are n_samples lists in list
    grp_top_set_syms_n_freq = []
    # grp_top_set_syms_n_freq without the frequency count
    grp_top_set_syms = []

    # lb_slices, e.g  [(483, 513, 523), (453, 513, 523), (393, 513, 523)],
    #  is one max_lookback_slice, e.g. (393, 513, 523), along with
    #  the remaining slices of the days_lookbacks,
    #  e.g. (483, 513, 523), (453, 513, 523)
    for i, lb_slices in enumerate(sets_lookback_slices):
        print(
            f"\n########## {i + 1} of {len(sets_lookback_slices)} lb_slices in sets_lookback_slices ##########"  # noqa
        )
        # unsorted list of the most frequent symbols in performance metrics
        #  of the lb_slices
        grp_most_freq_syms = []
        for j, lb_slice in enumerate(lb_slices):  # lb_slice, e.g. (246, 276, 286)
            iloc_start_train = lb_slice[0]  # iloc of start of training period
            iloc_end_train = lb_slice[1]  # iloc of end of training period
            iloc_start_eval = iloc_end_train  # iloc of start of evaluation period
            iloc_end_eval = lb_slice[2]  # iloc of end of evaluation period
            # one of the lookback-days in lookback-slice,
            #  e.g. 30 in [30, 60, 120]
            n_days = iloc_end_train - iloc_start_train
            # evaluation period for symbols picked in training
            d_eval = iloc_end_eval - iloc_start_eval

            _df = df_close.iloc[iloc_start_train:iloc_end_train]
            date_start_df_train = _df.index[0].strftime("%Y-%m-%d")
            date_end_df_train = _df.index[-1].strftime("%Y-%m-%d")
            len_lb_slices = len(lb_slices)

            if verbose:
                print(f"lb_slice {j+1} of {len_lb_slices}:           {lb_slice}")
                print(f"lb_slices:                 {lb_slices}")
                print(f"days eval:                 {d_eval}")
                print(f"iloc_start_train:          {iloc_start_train}")
                print(f"iloc_end_train:            {iloc_end_train}")
                print(f"date_start_df_train:       {date_start_df_train}")
                print(f"date_end_df_train:         {date_end_df_train}\n")

            # Example of perf_ranks:
            #     {'period-120': {'r_CAGR/UI': array(['CDNA', 'AVEO', 'DVAX',
            #     'XOMA', 'BLFS'], dtype=object),
            #     'r_CAGR/retnStd': array(['CDNA', 'AVEO', 'CYRX', 'BLFS',
            #     'XOMA'],
            #     dtype=object),
            #     'r_retnStd/UI': array(['GDOT', 'DVAX', 'XOMA', 'CTRA',
            #     'FTSM'],
            #     dtype=object)}}
            # Example of most_common_syms:
            #     [('XOMA', 3), ('CDNA', 2), ('AVEO', 2), ('DVAX', 2),
            #      ('BLFS', 2), ('CYRX', 1), ('GDOT', 1), ('CTRA', 1),
            #      ('FTSM', 1)]
            perf_ranks, most_freq_syms = rank_perf(_df, n_top_syms=n_top_syms)
            # unsorted list of the most frequent symbols in performance metric
            # of the lb_slices
            grp_most_freq_syms.append(most_freq_syms)
            if verbose:
                # print results of perf_ranks for period n_days
                # of perf. metric: r_CAGR/UI, r_CAGR/retnStd, r_retnStd/UI
                for period, value in perf_ranks.items():
                    print(f"perf_ranks for {period}:")
                    for perf_metric, sym_results in value.items():
                        # print(f"{perf_metric}: {sym_results}")
                        print(f"perf. metric: {perf_metric}")
                        print(f"best perf. symbols: {sym_results}")
                print("")
                # most common symbols of perf_ranks
                print(f"most_freq_syms: {most_freq_syms}")
                print(f"+++ finish lookback slice {lb_slice}, days={n_days} +++\n")

        if verbose:
            # grp_most_freq_syms a is list of lists of tuples of
            #  the most-common-symbols symbol:frequency cumulated from
            #  each days_lookback
            print(
                f"grp_most_freq_syms, len={len(grp_most_freq_syms)}: {grp_most_freq_syms}"
            )
            print(f"**** finish lookback slices {lb_slices} ****\n")

        # flatten list of lists of (symbol:frequency)
        flat_grp_most_freq_syms = [
            val for sublist in grp_most_freq_syms for val in sublist
        ]
        # return "symbol, frequency" pairs of the most frequent symbols,
        # i.e. best performing symbols, in flat_grp_most_freq_syms.
        # The paris are sorted in descending frequency.
        set_most_freq_syms = grp_tuples_sort_sum(flat_grp_most_freq_syms, reverse=True)
        # get the top n_top_syms of the most frequent "symbol, frequency" pairs
        top_set_syms_n_freq = set_most_freq_syms[0:n_top_syms]
        # get symbols from top_set_syms_n_freq,
        # i[0] = symbol, i[1]=symbol's frequency count
        top_set_syms = [i[0] for i in top_set_syms_n_freq[syms_start:syms_end]]

        # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq,
        # e.g. [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
        # [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
        # where each list is the best performing symbols from a lb_slices,
        # e.g. [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
        grp_top_set_syms_n_freq.append(top_set_syms_n_freq)
        grp_top_set_syms.append(top_set_syms)

        if verbose:
            print(
                f"top {n_top_syms} ranked symbols and frequency from set {lb_slices}:\n{top_set_syms_n_freq}"  # noqa
            )
            print(
                f"top {n_top_syms} ranked symbols from set {lb_slices}, sliced [{syms_start}:{syms_end}]:\n{top_set_syms}"  # noqa
            )
            print(
                f"===== finish top {n_top_syms} ranked symbols from days_lookback set {lb_slices} =====\n\n"  # noqa
            )

    return grp_top_set_syms_n_freq, grp_top_set_syms, date_end_df_train
