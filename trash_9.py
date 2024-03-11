

def get_grp_top_syms_n_freq_v1(df, list_random_lookback_slices, days_lookbacks, n_top_syms, syms_start, syms_end, verbose=False):
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


