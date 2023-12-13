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
