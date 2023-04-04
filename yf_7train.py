import pandas as pd
from myUtils import pickle_load, pickle_dump
from yf_utils import split_train_val_test, random_slices, lookback_slices
from yf_utils import top_set_sym_freq_cnt, get_grp_top_syms_n_freq
from yf_utils import eval_grp_top_syms_n_freq


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 16)
pd.set_option("display.width", 140)  # code-runner format

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
    df, max_lookback_slices, sets_lookback_slices, grp_top_set_syms_n_freq, days_lookbacks, days_eval, n_samples, n_top_syms, syms_start, syms_end, verbose
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
