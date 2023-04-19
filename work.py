def calc_portf_value_date_n(my_dates, my_symbols, df_close, my_portf_shares, n, verbose=False):
  
  z_date_syms_shares = zip(my_dates, my_symbols, my_portf_shares)

  date_exec = []  # buy date of portfolio
  shares_syms = []  # lists of shares of each symbol brought on date
  value_portf = [] # list of porfolio value on date
  # shares_SPY = []  # list of shares of SPY brought on date
  # value_SPY = []  # list of value of SPY shares on date 

  for date, symbols, portf_shares in z_date_syms_shares:
    next_date_n = get_NYSE_date_n(date, dates_NYSE_2013_2023, n, verbose=False)
    close_date_n = is_date_in_close(next_date_n, df_close)

    if close_date_n is None:
      p_date_exec = None
      p_ar_shares = None
      p_value_portf = None  # set to None when data are not available in df_close
      # SPY_shares = None
      # SPY_value = None  # set to None when data are not available in df_close

      if verbose:
        print(f"No data for close_date_n {close_date_n}, pick's portf value = None")
        # print(f'No data for next_date_n {next_date_n}, SPY portf value =    None')

    else: 
      if portf_shares is None:
        p_date_exec = None
        p_ar_shares = None
        p_value_portf = None

      else:  
        # p_ar_shares = calc_portf_shares(df_close, next_date_n, symbols, portf_target)
        p_date_exec, p_ar_syms, p_ar_price, p_ar_shares, p_ar_value, p_value_portf = \
          calc_portf_value(df_close, close_date_n, symbols, portf_shares, verbose)
          # calc_portf_value(df_close, next_date_n, symbols, portf_shares, verbose)      

        if verbose:
          print(f"next_date_n pick's portf value = {p_value_portf}")
          # print(f'next_date_n SPY portf value =    {SPY_value}')

    date_exec.append(p_date_exec)
    shares_syms.append(p_ar_shares)
    value_portf.append(p_value_portf)
    # shares_SPY.append(SPY_shares)
    # value_SPY.append(SPY_value)

    print('='*20, '\n')

  return date_exec, shares_syms, value_portf   
    