# def best_perf_syms_sets_lookback_slices(sets_lookback_slices, verbose=False):
#     # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
#     #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
#     #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
#     #   where each list is the best performing symbols from a lb_slices, e.g.
#     #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
#     grp_top_set_syms_n_freq = (
#         []
#     )  # list of lists of top_set_symbols_n_freq, there are n_samples lists in list
#     grp_top_set_syms = []  # grp_top_set_syms_n_freq without the frequency count

#     # lb_slices, e.g  [(483, 513, 523), (453, 513, 523), (393, 513, 523)],
#     #  is one max_lookback_slice, e.g. (393, 513, 523), along with
#     #  the remaining slices of the days_lookbacks, e.g. (483, 513, 523), (453, 513, 523)
#     for i, lb_slices in enumerate(sets_lookback_slices):
#         print(
#             f"\n########## {i + 1} of {len(sets_lookback_slices)} lb_slices in sets_lookcak_slices ##########"
#         )
#         # unsorted list of the most frequent symbols in performance metrics of the lb_slices
#         grp_most_freq_syms = []
#         for j, lb_slice in enumerate(lb_slices):  # lb_slice, e.g. (246, 276, 286)
#             iloc_start_train = lb_slice[0]  # iloc of start of training period
#             iloc_end_train = lb_slice[1]  # iloc of end of training period
#             iloc_start_eval = iloc_end_train  # iloc of start of evaluation period
#             iloc_end_eval = lb_slice[2]  # iloc of end of evaluation period
#             n_days = (
#                 iloc_end_train - iloc_start_train
#             )  # one of the lookback-days in lookback-slice
#             d_eval = iloc_end_eval - iloc_start_eval

#             _df = df.iloc[iloc_start_train:iloc_end_train]
#             date_start_df_train = _df.index[0].strftime("%Y-%m-%d")
#             date_end_df_train = _df.index[-1].strftime("%Y-%m-%d")

#             if verbose:
#                 print(
#                     f"days lookback:       {n_days},  {j + 1} of {len(days_lookbacks)} days_lookbacks: {days_lookbacks}"
#                 )
#                 print(f"lb_slices:           {lb_slices}")
#                 print(f"lb_slice:            {lb_slice}")
#                 print(f"days eval:           {d_eval}")
#                 print(f"iloc_start_train:    {iloc_start_train}")
#                 print(f"iloc_end_train:      {iloc_end_train}")
#                 print(f"date_start_df_train: {date_start_df_train}")
#                 print(f"date_end_df_train:   {date_end_df_train}")

#             perf_ranks, most_freq_syms = rank_perf(_df, n_top_syms=n_top_syms)
#             # unsorted list of the most frequent symbols in performance metrics of the lb_slices
#             grp_most_freq_syms.append(most_freq_syms)
#             if verbose:
#                 # n_days of r_CAGR/UI, r_CAGR/retnStd, r_retnStd/UI
#                 print(f"perf_ranks: {perf_ranks}")
#                 # most common symbols of perf_ranks
#                 print(f"most_freq_syms: {most_freq_syms}")
#                 print(f"+++ finish lookback slice {n_days} +++\n")

#         if verbose:
#             print(f"grp_most_freq_syms: {grp_most_freq_syms}")
#             # grp_most_freq_syms a is list of lists of tuples of
#             #  the most-common-symbols symbol:frequency cumulated from
#             #  each days_lookback
#             print(f"**** finish lookback slices {lb_slices} ****\n")

#         # flatten list of lists of (symbol:frequency)
#         flat_grp_most_freq_syms = [
#             val for sublist in grp_most_freq_syms for val in sublist
#         ]
#         # return "symbol, frequency" pairs of the most frequent symbols, i.e. best performing symbols,
#         #  in flat_grp_most_freq_syms. The paris are sorted in descending frequency.
#         set_most_freq_syms = grp_tuples_sort_sum(flat_grp_most_freq_syms, reverse=True)
#         # get the top n_top_syms of the most frequent "symbol, frequency" pairs
#         top_set_syms_n_freq = set_most_freq_syms[0:n_top_syms]
#         # get symbols from top_set_syms_n_freq, i[0] = symbol, i[1]=symbol's frequency count
#         top_set_syms = [i[0] for i in top_set_syms_n_freq[syms_start:syms_end]]

#         # grp_top_set_syms_n_freq is a list of lists of top_set_syms_n_freq, e.g.
#         #   [[('AGY', 7), ('PCG', 7), ('KDN', 6), ..., ('CYT', 3)],
#         #    [('FCN', 9), ('HIG', 9), ('SJR', 8), ..., ('BFH', 2)]]
#         #   where each list is the best performing symbols from a lb_slices, e.g.
#         #     [(483, 513, 523), (453, 513, 523), (393, 513, 523)]
#         grp_top_set_syms_n_freq.append(top_set_syms_n_freq)
#         grp_top_set_syms.append(top_set_syms)

#         if verbose:
#             print(
#                 f"top {n_top_syms} ranked symbols and frequency from set {lb_slices}:\n{top_set_syms_n_freq}"
#             )
#             print(
#                 f"top {n_top_syms} ranked symbols from set {lb_slices}:\n{top_set_syms}"
#             )
#             print(
#                 f"===== finish top {n_top_syms} ranked symbols from days_lookback set {lb_slices} =====\n\n"
#             )

#     return grp_top_set_syms_n_freq, grp_top_set_syms, date_end_df_train


import pandas as pd
from yf_utils import random_slices, lookback_slices
from yf_utils import rank_perf, grp_tuples_sort_sum, top_set_sym_freq_cnt
from myUtils import pickle_load, pickle_dump


pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 11)
pd.set_option("display.max_colwidth", 16)
pd.set_option("display.width", 145)

path_dir = "C:/Users/ping/MyDrive/stocks/yfinance/"
path_data_dump = path_dir + "VSCode_dump/"

fp_df_close_clean = "df_close_clean"

### Set parameters ######################################################################
# fp_df_eval_results = f'df_eval_results_{run_type}'
fp_df_picks = f"df_picks"

# verbose = False  # True prints more output
verbose = True  # True prints more output

store_results = True
# store_results = False

n_samples = 1  # only need 1 sample to get the current picks

# for training, the number of days to lookback from iloc max-lookback iloc_end_train
l_days_lookbacks = [[30, 60, 120], [15, 30, 60, 120]]

l_sorted_days_lookbacks = []
for days_lookbacks in l_days_lookbacks:
    l_sorted_days_lookbacks.append(sorted(days_lookbacks))

# number of days from iloc_end_train are used to evaluate effectiveness of the training
days_eval = 0

# number of the most-common symbols from days_lookbacks' performance rankings to keep
n_top_syms = 20

# slice starts and ends for selecting the best performing symbols
syms_start = 0
syms_end = 10


#########################################################################



print(f"verbose : {verbose }")
print(f"store_results: {store_results}")
print(f"n_samples: {n_samples}")
print(f"days_lookbacks: {days_lookbacks}")
print(f"days_eval: {days_eval}")
print(f"n_top_syms: {n_top_syms}")
print(f"syms_start: {syms_start}")
print(f"syms_end: {syms_end}")
print(f"fp_df_picks: {fp_df_picks}\n\n")


fp_df_close_clean = "df_close_clean"
fp_dates_missing_in_df_picks = "dates_missing_in_df_picks"

df_picks = pickle_load(path_data_dump, fp_df_picks)
df_close_clean = pickle_load(path_data_dump, fp_df_close_clean)
df_current = df_close_clean.copy()

# get picks of previous days by dropping the last n rows from df_current
#  drop_last_n_rows = 1 drops the last row from df_current
drop_last_n_rows = 0
# drop_last_n_rows = 4

dates_missing_in_df_picks = \
  pickle_load(path_data_dump, fp_dates_missing_in_df_picks)
print(
    f"dates_missing_in_df_picks, len({len(dates_missing_in_df_picks)}):\n{dates_missing_in_df_picks}"
)

get_n_dates_to_drop = get_n_dates_to_drop(df_current.index, dates_missing_in_df_picks)
print(get_n_dates_to_drop)



# for i, days_lookbacks in enumerate(l_sorted_days_lookbacks):

#     print(f'\n\n{"%"*40:<42}{i+1} of {len(l_sorted_days_lookbacks)} days_lookbacks{"%"*40:>42}\n')
#     max_days_lookbacks = max(days_lookbacks)
#     print(f"max_days_lookbacks: {max_days_lookbacks}\n")

#     slice_start = -(max_days_lookbacks + drop_last_n_rows)
#     slice_end = -drop_last_n_rows
#     if drop_last_n_rows == 0:  # return df with all rows
#         df = df_current[slice_start:].copy()
#     else:  # return df with dropped drop_last_n_rows rows
#         df = df_current[slice_start:slice_end].copy()


#     print(f"df.head():\n{df.head()}\n")        
#     print(f"dropped last {drop_last_n_rows} row(s) from df")
#     print(f"df.tail():\n{df.tail()}\n")
#     len_df = len(df)
#     # len_df_current = len(df_current)
#     print(f"len(df): {len(df)}")




    # # return n_samples slices
    # max_lookback_slices = random_slices(
    #     len_df,
    #     n_samples=n_samples,
    #     days_lookback=max(days_lookbacks),
    #     days_eval=days_eval,
    #     verbose=False,
    # )
    # # return n_samples * len(days_lookbacks) slices
    # sets_lookback_slices = lookback_slices(
    #     max_slices=max_lookback_slices, days_lookbacks=days_lookbacks, verbose=False
    # )

    # if verbose:
    #     print(f"number of max_lookback_slices is equal to n_samples = {n_samples}")
    #     print(f"max_lookback_slices:\n{max_lookback_slices}\n")
    #     print(f"days_lookbacks: {days_lookbacks}")
    #     print(f"sets_lookback_slices:\n{sets_lookback_slices}\n")
    #     print(f"number of sets in sets_lookback_slices is equal to n_samples = {n_samples}")
    #     print(
    #         f'number of tuples in each "set of lookback slices" is equal to len(days_lookbacks): {len(days_lookbacks)}'
    #     )

    # (
    #     grp_top_set_syms_n_freq,
    #     grp_top_set_syms,
    #     date_end_df_train,
    # ) = best_perf_syms_sets_lookback_slices(sets_lookback_slices, verbose=verbose)

    # for i, top_set_syms_n_freq in enumerate(grp_top_set_syms_n_freq):
    #     l_sym_freq_cnt = top_set_sym_freq_cnt(top_set_syms_n_freq)
    #     if verbose:
    #         print(f"set_lookback_slices: {sets_lookback_slices[i]}")
    #         print(f"max_lookback_slices: {max_lookback_slices}\n")
    #         print(f"data below will be added to {fp_df_picks}")
    #         print(f"date_end_df_train:   {date_end_df_train}")
    #         print(f"max_days_lookbacks:  {max_days_lookbacks}")
    #         print(f"days_lookbacks:      {days_lookbacks}")
    #         print(f"sym_freq_15:         {l_sym_freq_cnt[0]}")
    #         print(f"sym_freq_14:         {l_sym_freq_cnt[1]}")
    #         print(f"sym_freq_13:         {l_sym_freq_cnt[2]}")
    #         print(f"sym_freq_12:         {l_sym_freq_cnt[3]}")
    #         print(f"sym_freq_11:         {l_sym_freq_cnt[4]}")
    #         print(f"sym_freq_10:         {l_sym_freq_cnt[5]}")
    #         print(f"sym_freq_9:          {l_sym_freq_cnt[6]}")
    #         print(f"sym_freq_8:          {l_sym_freq_cnt[7]}")
    #         print(f"sym_freq_7:          {l_sym_freq_cnt[8]}")
    #         print(f"sym_freq_6:          {l_sym_freq_cnt[9]}")
    #         print(f"sym_freq_5:          {l_sym_freq_cnt[10]}")
    #         print(f"sym_freq_4:          {l_sym_freq_cnt[11]}")
    #         print(f"sym_freq_3:          {l_sym_freq_cnt[12]}")
    #         print(f"sym_freq_2:          {l_sym_freq_cnt[13]}\n")

    # if store_results:
    #     row_picks0 = [date_end_df_train, max_days_lookbacks, str(days_lookbacks)]
    #     row_picks1 = [
    #         str(l_sym_freq_cnt[0]),
    #         str(l_sym_freq_cnt[1]),
    #         str(l_sym_freq_cnt[2]),
    #         str(l_sym_freq_cnt[3]),
    #     ]
    #     row_picks2 = [
    #         str(l_sym_freq_cnt[4]),
    #         str(l_sym_freq_cnt[5]),
    #         str(l_sym_freq_cnt[6]),
    #         str(l_sym_freq_cnt[7]),
    #     ]
    #     row_picks3 = [
    #         str(l_sym_freq_cnt[8]),
    #         str(l_sym_freq_cnt[9]),
    #         str(l_sym_freq_cnt[10]),
    #         str(l_sym_freq_cnt[11]),
    #     ]
    #     row_picks4 = [str(l_sym_freq_cnt[12]), str(l_sym_freq_cnt[13])]
    #     row_picks_total = row_picks0 + row_picks1 + row_picks2 + row_picks3 + row_picks4
    #     print(f"row_picks_total: {row_picks_total}")

    #     df_picks.loc[len(df_picks)] = row_picks_total
    #     pickle_dump(df_picks, path_data_dump, fp_df_picks)
    #     print(f"appended row_picks_total to df_picks:\n{row_picks_total}\n")
