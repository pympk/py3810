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


def get_grp_top_syms_n_freq(df, sets_lookback_slices, verbose=False):
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
    for i, lb_slices in enumerate(sets_lookback_slices):
        # print(f'\n########## {i + 1} of {len(sets_lookback_slices)} lb_slices in sets_lookcak_slices ##########')
        print(
            f"\n########## {i + 1} of {len(sets_lookback_slices)} lb_slices: {lb_slices} in sets_lookcak_slices ##########"
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
            perf_ranks, most_freq_syms = _5_perf_ranks(_df, n_top_syms=n_top_syms)
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
        set_most_freq_syms = _6_grp_tuples_sort_sum(
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


def eval_grp_top_syms_n_freq(df, max_lookback_slices, grp_top_set_syms_n_freq, verbose):
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

        if run_type != "current":
            idx_eval = df.index[iloc_start_eval:iloc_end_eval]
            date_start_df_eval = idx_eval[0].strftime("%Y-%m-%d")
            date_end_df_eval = idx_eval[-1].strftime("%Y-%m-%d")
            print(f"len(idx_eval): {len(idx_eval)}")
            print(f"idx_eval: {idx_eval}")
            print(
                f"df_eval dates (inclusive): {date_start_df_eval} - {date_end_df_eval}\n"
            )
        else:  # run_type == "current'
            print(f"run_type: {run_type}, idx_eval is empty: {idx_eval.empty}")

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
            if run_type == "current":
                msg_stop = f'\n\n**** Normal termination for run_type "{run_type}". Wrote current picks to df_picks ****'
                print(f"df_picks:\n{df_picks}")
                raise SystemExit(msg_stop)
            else:
                msg_stop = f'\n\n**** STOPPED df_eval is empty but run_type != "current", the run_type is "{run_type}" ****'
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


import pandas as pd
import datetime
from yf_utils import _2_split_train_val_test, _3_random_slices, _4_lookback_slices
from yf_utils import _5_perf_ranks, _6_grp_tuples_sort_sum, top_set_sym_freq_cnt
from myUtils import pickle_load, pickle_dump

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 16)
pd.set_option("display.width", 790)

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"

fp_df_close_clean = "df_close_clean"


# #### Select run parameters. Parameters can also be passed using papermill by running yf_7_freq_cnt_pm_.ipynb


# SELECT RUN PARAMETERS.async Parameters can also be passed using papermill by running yf_7_freq_cnt_pm_.ipynb
verbose = True  # True prints more output
# verbose = False  # True prints more output

# write run results to df_eval_results
# store_results = False
store_results = True

# select run type
# run_type = 'train'
# run_type = 'validate'
run_type = "test"
# run_type = 'current'

# number of max lookback tuples to create for iloc iloc_start_train:iloc_end_train:iloc_end_eval
# i.e. number of grp_top_set_syms_n_freq and grp_top_set_syms
# n_samples = 400
# n_samples = 20
n_samples = 2

# for training, the number of days to lookback from iloc max-lookback iloc_end_train
days_lookbacks = [15, 30, 60]
# days_lookbacks = [30, 60, 120]
# days_lookbacks = [15, 30, 60, 120]
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

# get picks of previous days by dropping the last n rows from df_current
#  drop_last_n_rows = 1 drops the last row from df_current
drop_last_n_rows = 0

# over-ride parameters for run_type 'current'
if run_type == "current":
    days_eval = 0  # no need to eval, df_eval will be empty
    n_samples = 1  # no need to repeat sample the current result, repeat sample will yield the same tuple


##########################################################
fp_df_eval_results = f"df_eval_results_{run_type}"
# fp_df_eval_results = f'df_eval_trash_new'
##########################################################


fp_df_picks = f"df_picks"

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
print(f"fp_df_picks: {fp_df_picks}")


if run_type != "current":
    df_eval_results = pickle_load(path_data_dump, fp_df_eval_results)

df_picks = pickle_load(path_data_dump, fp_df_picks)
df_close_clean = pickle_load(path_data_dump, fp_df_close_clean)


# #### Split dataframe into Train, Validate and Test


# Split df_close_clean into training (df_train), validation (df_val) and test (df_test) set.
# The default split is 0.7, 0.2, 0.1 respectively.
df_train, df_val, df_test = _2_split_train_val_test(df_close_clean)


# #### Create df_current, a df with the latest data


max_days_lookbacks = max(days_lookbacks)
print(f"max_days_lookbacks: {max_days_lookbacks}")


# df_current = df_test.tail(max_days_lookbacks)
df_current = df_test.copy()

# df_current.tail(3)


# #### Load df according to run_type


if run_type == "train":
    df = df_train.copy()
elif run_type == "validate":
    df = df_val.copy()
elif run_type == "test":
    df = df_test.copy()
elif run_type == "current":  # get the current picks
    print(f"run_type: current")
    slice_start = -(max_days_lookbacks + drop_last_n_rows)
    slice_end = -drop_last_n_rows
    if drop_last_n_rows == 0:  # return df with all rows
        df = df_current[slice_start:].copy()
    else:  # return df with dropped drop_last_n_rows rows
        df = df_current[slice_start:slice_end].copy()
    print(f"dropped last {drop_last_n_rows} row(s) from df")
    print(f"df.tail():\n{df.tail()}\n")
else:
    msg_stop = f"ERROR run_type must be 'train', 'validate', 'test' or 'current', run_type is: {run_type}"
    raise SystemExit(msg_stop)


# #### Print dataframe for the run, and lengths of other dataframes


print(f"run_type: {run_type}, df.tail(3):\n{df.tail(3)}\n")
len_df = len(df)
len_df_train = len(df_train)
len_df_val = len(df_val)
len_df_test = len(df_test)
len_df_current = len(df_current)
print(f"run_type: {run_type}, len(df): {len(df)}")
print(
    f"len_df_train: {len_df_train}, len_df_val: {len_df_val}, len_df_test: {len_df_test}, len_df_current: {len_df_current} "
)


# #### Create a sets of iloc lookback slices (iloc_start_train:iloc_end_train:iloc_end_eval), where:
# * iloc_end_train - iloc_start_train = days_lookback
# * iloc_end_eval - iloc_end_train = days_eval
# #### for example, if given:
# * n_samples = 2
# * days_lookbacks = [10, 20, 30]
# * days_eval = 5
# #### a possible result is:
#   - max_lookback_slices: [(417, 447, 452), (265, 295, 300)], where:
#     - len(max_lookback_slices) = n_samples = 2
#     - middle number in the tuples, 447 and 295, is the iloc of the "pivot day" for the days in "days_lookbacks" to lookback
#     - 447 - 417 = middle number - first number = max(days_lookbacks) = 30
#     - 295 - 265 = middle number - first number = max(days_lookbacks) = 30
#     - 452 - 447 = last number - middle number = days_eval = 5
#     - 300 - 295 = last number - middle number = days_eval = 5
#   - sets_lookback_slices: [[(437, 447, 452), (427, 447, 452), (417, 447, 452)], [(285, 295, 300), (275, 295, 300), (265, 295, 300)]], where:
#     - len(sets_lookback_slices) = n_samples = 2
#     - last tuple in each list, i.e. (417, 447, 452) and (265, 295, 300), is a tuple from max_lookback_slices
#     - where a set, e.g. [(437, 447, 452), (427, 447, 452), (417, 447, 452)]:
#       - middle number, 447, iloc of the "pivot day" is constant for that set
#       - middle number - first number, is the training period specified in days_lookbacks
#         - 447 - 437 = middle number - first number = days_lookbacks[0] = 10
#         - 447 - 427 = middle number - first number = days_lookbacks[1] = 20
#         - 447 - 417 = middle number - first number = days_lookbacks[2] = 30
#       - last number, 452, iloc of the end of the evaluation period is constant
#         - 452 - 447 = last number - middle number = days_eval = 5


# return n_samples slices
max_lookback_slices = _3_random_slices(
    len_df,
    n_samples=n_samples,
    days_lookback=max(days_lookbacks),
    days_eval=days_eval,
    verbose=False,
)
# return n_samples * len(days_lookbacks) slices
sets_lookback_slices = _4_lookback_slices(
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


# #### Relationship between max_lookbak_slices, sets_lookback_slices and days_lookback
# ![](images\fig_rel_max_lb_sets_lb_d_lb.jpg)


# #### Generate grp_top_set_syms_n_freq. It is a list of sub-lists, e.g.:
#  - [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)], ..., [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
# #### grp_top_set_syms_n_freq has n_samples sub-lists. Each sub-list corresponds to a tuple in the max_lookback_slices. Each sub-list has n_top_syms tuples of (symbol, frequency) pairs, and is sorted in descending order of frequency. The frequency is the number of times the symbol appears in the top n_top_syms performance rankings of CAGR/UI, CAGR/retnStd and retnStd/UI.
# #### Therefore, symbols in the sub-list are the best performing symbols for the periods in days_lookbacks. Each sub-list corresponds to a tuple in max_lookback_slices. There are as many sub-lists as there are tuples in max_lookback_slices.


grp_top_set_syms_n_freq, grp_top_set_syms, dates_end_df_train = get_grp_top_syms_n_freq(
    df, sets_lookback_slices, verbose
)


grp_top_set_syms_n_freq


# #### print the best performing symbols for each set in sets_lookback_slices


for i, top_set_syms_n_freq in enumerate(grp_top_set_syms_n_freq):
    l_sym_freq_cnt = top_set_sym_freq_cnt(top_set_syms_n_freq)
    if verbose:
        print(f"max_lookback_slices:             {max_lookback_slices}")
        # print(f'set_lookback_slices: {sets_lookback_slices[i]}\n')
        print(
            f"set_lookback_slices {i + 1} of {len(sets_lookback_slices):>3}:    {sets_lookback_slices[i]}\n"
        )

        if run_type == "current":
            print(f"data below will be added to      {fp_df_picks}")

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

if run_type == "current":  # record results to df
    row_picks0 = [date_end_df_train, max_days_lookbacks, str(days_lookbacks)]
    row_picks1 = [
        str(l_sym_freq_cnt[0]),
        str(l_sym_freq_cnt[1]),
        str(l_sym_freq_cnt[2]),
        str(l_sym_freq_cnt[3]),
    ]
    row_picks2 = [
        str(l_sym_freq_cnt[4]),
        str(l_sym_freq_cnt[5]),
        str(l_sym_freq_cnt[6]),
        str(l_sym_freq_cnt[7]),
    ]
    row_picks3 = [
        str(l_sym_freq_cnt[8]),
        str(l_sym_freq_cnt[9]),
        str(l_sym_freq_cnt[10]),
        str(l_sym_freq_cnt[11]),
    ]
    row_picks4 = [str(l_sym_freq_cnt[12]), str(l_sym_freq_cnt[13])]
    row_picks_total = row_picks0 + row_picks1 + row_picks2 + row_picks3 + row_picks4
    print(f"row_picks_total: {row_picks_total}")

    df_picks.loc[len(df_picks)] = row_picks_total
    pickle_dump(df_picks, path_data_dump, fp_df_picks)
    print(f"appended row_picks_total to df_picks:\n{row_picks_total}\n")


# #### Evaluate performance of symbols in set_lookback_slices versus SPY


l_row_add_total = eval_grp_top_syms_n_freq(
    df, max_lookback_slices, grp_top_set_syms_n_freq, verbose
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
