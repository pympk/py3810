import pandas as pd
import numpy as np
from myUtils import pickle_load, pickle_dump
from myUtils import symb_perf_stats_vectorized_v8 as sps
from yf_utils import split_train_val_test, random_slices, lookback_slices
from yf_utils import top_set_sym_freq_cnt, get_grp_top_syms_n_freq






def compare_picks_v_SPY(df_eval, df_SPY):
    # import numpy as np
    # from myUtils import symb_perf_stats_vectorized_v8 as sps

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

def eval_grp_top_syms_n_freq(df, max_lookback_slices, sets_lookback_slices, grp_top_set_syms_n_freq, verbose):
# def eval_grp_top_syms_n_freq(df, max_lookback_slices, sets_lookback_slices, grp_top_set_syms_n_freq, days_lookbacks, days_eval, n_samples,  n_top_syms, syms_start, syms_end, verbose):
    # import pandas as pd
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
            # print(f"days_lookbacks:      {days_lookbacks}")
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



if __name__ == '__main__':

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.max_columns", 11)
    pd.set_option("display.max_colwidth", 10)
    pd.set_option("display.width", 160)  # code-runner format

    path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
    path_data_dump = path_dir + "VSCode_dump/"

    fp_df_close_clean = "df_close_clean"

    #######################################################################
    ## SELECT RUN PARAMETERS.async Parameters can also be passed using papermill by running yf_7_freq_cnt_pm_.ipynb
    # verbose = True  # True prints more output
    verbose = False  # True prints more output

    # write run results to df_eval_results
    # store_results = False
    store_results = True

    # select run type
    run_type = "train"
    # run_type = 'validate'
    # run_type = "test"

    # number of max lookback tuples to create for iloc iloc_start_train:iloc_end_train:iloc_end_eval
    # i.e. number of grp_top_set_syms_n_freq and grp_top_set_syms
    # n_samples = 400
    n_samples = 2

    # for training, the number of days to lookback from iloc max-lookback iloc_end_train
    # days_lookbacks = [15, 30, 60]
    # days_lookbacks = [30, 60, 120]
    days_lookbacks = [15, 30, 60, 120]
    days_lookbacks.sort()

    # number of days in dataframe to evaluate effectiveness of the training, days_eval = len(df_eval)
    #  days_eval = 4 means buy at close on 1st day after the signal, hold for 2nd and 3rd day, sell at close on 4th day
    days_eval = 4
    # days_eval = 5

    # number of the most-common symbols from days_lookbacks' performance rankings to keep
    n_top_syms = 20
    # n_top_syms = 5

    # slice starts and ends for selecting the best performing symbols
    syms_start = 0
    syms_end = 10


    ##########################################################
    # fp_df_eval_results = f"df_eval_results_{run_type}"
    fp_df_eval_results = "df_eval_trash"
    ##########################################################
    #######################################################################


    print(f"verbose : {verbose }")
    print(f"store_results: {store_results}")
    print(f"run_type: {run_type}")
    print(f"n_samples: {n_samples}")
    print(f"days_lookbacks: {days_lookbacks}")
    print(f"days_eval: {days_eval}")
    print(f"n_top_syms: {n_top_syms}")
    print(f"syms_start: {syms_start}")
    print(f"syms_end: {syms_end}")
    print(f"fp_df_eval_results: {fp_df_eval_results}")

    df_close_clean = pickle_load(path_data_dump, fp_df_close_clean)

    # Split df_close_clean into training (df_train), validation (df_val) and test (df_test) set.
    # The default split is 0.7, 0.2, 0.1 respectively.
    df_train, df_val, df_test = split_train_val_test(df_close_clean)

    max_days_lookbacks = max(days_lookbacks)
    print(f"max_days_lookbacks: {max_days_lookbacks}")

    # Load df according to run_type
    if run_type == "train":
        df = df_train.copy()
    elif run_type == "validate":
        df = df_val.copy()
    elif run_type == "test":
        df = df_test.copy()
    else:
        msg_stop = f"ERROR run_type must be 'train', 'validate', or 'test', run_type is: {run_type}"
        raise SystemExit(msg_stop)

    # Print dataframe for the run, and lengths of other dataframes
    print(f"run_type: {run_type}, df.tail(3):\n{df.tail(3)}\n")
    len_df = len(df)
    len_df_train = len(df_train)
    len_df_val = len(df_val)
    len_df_test = len(df_test)
    print(f"run_type: {run_type}, len(df): {len(df)}")
    print(
        f"len_df_train: {len_df_train}, len_df_val: {len_df_val}, len_df_test: {len_df_test}"
    )

    # return n_samples slices
    max_lookback_slices = random_slices(
        len_df,
        n_samples=n_samples,
        days_lookback=max(days_lookbacks),
        days_eval=days_eval,
        verbose=False,
    )
    # return n_samples * len(days_lookbacks) slices
    sets_lookback_slices = lookback_slices(
        max_slices=max_lookback_slices, days_lookbacks=days_lookbacks, verbose=False
    )

    if verbose:
        print(f"number of max_lookback_slices is equal to n_samples = {n_samples}")
        print(f"max_lookback_slices:\n{max_lookback_slices}\n")
        print(f"number of sets in sets_lookback_slices is equal to n_samples = {n_samples}")
        print(f"sets_lookback_slices:\n{sets_lookback_slices}\n")
        print(f"days_lookbacks: {days_lookbacks}")
        print(
            f'number of tuples in each "set of lookback slices" is equal to len(days_lookbacks): {len(days_lookbacks)}'
        )

    # #### Generate grp_top_set_syms_n_freq. It is a list of sub-lists, e.g.:
    #  - [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)], ..., [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
    # #### grp_top_set_syms_n_freq has n_samples sub-lists. Each sub-list corresponds to a tuple in the max_lookback_slices. Each sub-list has n_top_syms tuples of (symbol, frequency) pairs, and is sorted in descending order of frequency. The frequency is the number of times the symbol appears in the top n_top_syms performance rankings of CAGR/UI, CAGR/retnStd and retnStd/UI.
    # #### Therefore, symbols in the sub-list are the best performing symbols for the periods in days_lookbacks. Each sub-list corresponds to a tuple in max_lookback_slices. There are as many sub-lists as there are tuples in max_lookback_slices.
    grp_top_set_syms_n_freq, grp_top_set_syms, dates_end_df_train = get_grp_top_syms_n_freq(
        df, sets_lookback_slices, days_lookbacks, n_top_syms, syms_start, syms_end, verbose
    )

    # #### print the best performing symbols for each set in sets_lookback_slices
    for i, top_set_syms_n_freq in enumerate(grp_top_set_syms_n_freq):
        l_sym_freq_cnt = top_set_sym_freq_cnt(top_set_syms_n_freq)
        if verbose:
            print(f"max_lookback_slices:             {max_lookback_slices}")
            # print(f'set_lookback_slices: {sets_lookback_slices[i]}\n')
            print(
                f"set_lookback_slices {i + 1} of {len(sets_lookback_slices):>3}:    {sets_lookback_slices[i]}\n"
            )
            print(f"max_days_lookbacks:              {max_days_lookbacks}")
            print(f"df end date for days_lookbacks:  {dates_end_df_train[i]}")
            print(f"days_lookbacks:                  {days_lookbacks}")
            print(f"sym_freq_15:                     {l_sym_freq_cnt[0]}")
            print(f"sym_freq_14:                     {l_sym_freq_cnt[1]}")
            print(f"sym_freq_13:                     {l_sym_freq_cnt[2]}")
            print(f"sym_freq_12:                     {l_sym_freq_cnt[3]}")
            print(f"sym_freq_11:                     {l_sym_freq_cnt[4]}")
            print(f"sym_freq_10:                     {l_sym_freq_cnt[5]}")
            print(f"sym_freq_9:                      {l_sym_freq_cnt[6]}")
            print(f"sym_freq_8:                      {l_sym_freq_cnt[7]}")
            print(f"sym_freq_7:                      {l_sym_freq_cnt[8]}")
            print(f"sym_freq_6:                      {l_sym_freq_cnt[9]}")
            print(f"sym_freq_5:                      {l_sym_freq_cnt[10]}")
            print(f"sym_freq_4:                      {l_sym_freq_cnt[11]}")
            print(f"sym_freq_3:                      {l_sym_freq_cnt[12]}")
            print(f"sym_freq_2:                      {l_sym_freq_cnt[13]}\n")


    # #### Evaluate performance of symbols in set_lookback_slices versus SPY
    l_row_add_total = eval_grp_top_syms_n_freq(
        df, max_lookback_slices, sets_lookback_slices, grp_top_set_syms_n_freq, verbose
        # df, max_lookback_slices, sets_lookback_slices, grp_top_set_syms_n_freq, days_lookbacks, days_eval, n_samples, n_top_syms, syms_start, syms_end, verbose
    )

    for row_add_total in l_row_add_total:
        print(f"row_add_total: {row_add_total}")

    if store_results:  # record results to df
        df_eval_results = pickle_load(path_data_dump, fp_df_eval_results)
        print(f"df_eval_results BEFORW store results:\n{df_eval_results}\n")

        for row_add_total in l_row_add_total:
            # print(f'row_add_total: {row_add_total}')
            df_eval_results.loc[len(df_eval_results)] = row_add_total
            print(f"appended row_add_total to df_eval_results:\n{row_add_total}\n")

        pickle_dump(df_eval_results, path_data_dump, fp_df_eval_results)
        print(f"Save results to: {fp_df_eval_results}")
        df_eval_results = pickle_load(path_data_dump, fp_df_eval_results)
        print(f"df_eval_results AFTER store results:\n{df_eval_results}\n")




