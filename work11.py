def calc_portf_value_date_buy_(dates_in_days_lookbacks, my_symbols, df_close, portf_target, n, verbose=False):

  from ast import literal_eval
  from work11 import any_not_in_list, get_trading_date_n

  # Trading dates
  idx_close = df_close.index.strftime('%Y-%m-%d')

  # Symbols in df_close
  symbols_df_close = df_close.columns  # symbols in df_close   

  z_date_syms = zip(dates_in_days_lookbacks, my_symbols)

  date_exec = []  # buy date of portfolio
  shares_syms = []  # lists of shares of each symbol brought on date
  value_portf = [] # list of porfolio value on date
  shares_SPY = []  # list of shares of SPY brought on date
  value_SPY = []  # list of value of SPY shares on date 

  for date, syms in z_date_syms:
    next_date_n = get_trading_date_n(date, idx_close, n, verbose=False)
    close_date_n = is_date_in_close(next_date_n, df_close)



    print(f'++++++++++++++')
    print(f'next_date_n: {next_date_n}')
    print(f'close_date_n: {close_date_n}')

    l_syms = literal_eval(syms)  # convert list stored as str back to list
    # True if symbol(s) in l_syms is not a column in df_close


    sym_not_in_df_close = any_not_in_list(l_syms, symbols_df_close)
    # sym_not_in_df_close = any_not_in_list(l_syms, df_close.columns.tolist())


    if close_date_n is None or sym_not_in_df_close:

      print(f'l_syms: {l_syms}')  
      print(f'sym_not_in_df_close: {sym_not_in_df_close}')  
      print(f'++++++++++++++')




    # if close_date_n is None:
      p_date = None
      p_ar_shares = None
      p_portf_value = None  # set to None when data are not available in df_close
      SPY_shares = None
      SPY_value = None  # set to None when data are not available in df_close

      if verbose:
        print(f"No data for close_date_n {close_date_n}, pick's portf value = None")
        print(f'No data for close_date_n {close_date_n}, SPY portf value =    None')

    else:    
      p_ar_shares = calc_portf_shares(df_close, close_date_n, syms, portf_target)
      p_date, l_syms, ar_price, ar_shares, ar_value, p_portf_value = \
        calc_portf_value(df_close, close_date_n, syms, p_ar_shares, verbose)

      syms = str(['SPY'])
      SPY_shares = calc_portf_shares(df_close, close_date_n, syms, portf_target)
      date, l_syms, ar_price, ar_shares, ar_value, SPY_value = \
        calc_portf_value(df_close, close_date_n, syms, SPY_shares, verbose)

      if verbose:
        print(f"close_date_n pick's portf value = {p_portf_value}")
        print(f'close_date_n SPY portf value =    {SPY_value}')

    date_exec.append(p_date)
    shares_syms.append(p_ar_shares)
    value_portf.append(p_portf_value)
    shares_SPY.append(SPY_shares)
    value_SPY.append(SPY_value)

    print('='*20, '\n')

  return date_exec, shares_syms, value_portf, shares_SPY, value_SPY       


def get_trading_date_n(date, idx_close, n, verbose=False):
  """
  This function takes a date, a closing price date index, a number of days (n), and an optional verbosity flag,
  and returns the trading date n days away from the input date if it exists in the index, or None otherwise.

  Args:
    date: The date to start from (as a datetime object).
    idx_close: A dictionary or Series containing date index of close prices.
    n: The number of days to move forward or backward in time (positive for future, negative for past).
    verbose: An optional flag to print additional information about the process (default: False).

  Returns:
    The trading date n days away from the input date if it exists in the index, or None otherwise.
  """

  # Check if the input date is present in the closing price index.
  if date in idx_close:
    # Get the last and current index positions of the input date.
    idx_last = len(idx_close) - 1  # Last index of the closing price index.
    idx_date = idx_close.get_loc(date)  # Index of the input date in the index.

    # Calculate the index of the date n days away from the input date.
    idx_date_n = idx_date + n

    # Print debug information if verbose flag is set.
    if verbose:
      print(f"date: {date} is in idx_close, "
            f"date's position in idx_close is: {idx_date} of {idx_last}, "
            f"n: {n}, idx_date_n: {idx_date_n},")

    # Check if the calculated index is within the bounds of the closing price index.
    if 0 <= idx_date_n <= idx_last:
      # Get the date n days away from the input date using the calculated index.
      date_n = idx_close[idx_date_n]

      # Print debug information if verbose flag is set.
      if verbose:
        print(f"idx_date_n: {idx_date_n} is within bounds of idx_close (0 to {idx_last}), date_n: {date_n}\n")

    else:
      # If the calculated index is out of bounds, set the output date to None.
      date_n = None

      # Print debug information if verbose flag is set.
      if verbose:
        print(f"idx_date_n: {idx_date_n} is out-of-bounds of idx_close (0 to {idx_last})\n")

  else:
    # If the input date is not in the closing price index, set the output date to None.
    date_n = None

    # Print debug information if verbose flag is set.
    if verbose:
      print(f"date: {date} is not in idx_close\n")

  # Return the date n days away from the input date if it exists in the index, or None otherwise.
  return date_n


def any_not_in_list(list1, list2):
  """
  Checks if any items in list1 are not in list2.

  Args:
    list1: A list of items.
    list2: Another list of items.

  Returns:
    True if any item in list1 is not in list2, False if all are present.
  """
  return bool(set(list1) - set(list2))

def is_date_in_close(date, df_close):
  idx_close = df_close.index.strftime('%Y-%m-%d')
  if date in idx_close:
    return date
  else:
    return None
  
def calc_portf_shares(df_close, date, str_symbols, portf_target):
  # calculate number of shares to buy for symbols in str_symbols to meet port_target
  import numpy as np
  from ast import literal_eval    
  l_syms = literal_eval(str_symbols)  # convert list stored as str back to list
  # array of closing prices corresponding to symbols in l_syms    
  ar_price = df_close.loc[date][l_syms].values  
  sym_cnt = len(l_syms)  # number of symbols
  amt_per_sym = portf_target / sym_cnt  # target dollar investment in each symbol
  ar_shares = np.floor(amt_per_sym / ar_price)  # array of shares for each symbol
  return ar_shares


def calc_portf_value(df_close, date, str_symbols, ar_shares, verbose=False):
    import numpy as np
    from ast import literal_eval    
    l_syms = literal_eval(str_symbols)  # convert list stored as str back to list
    # array of closing prices corresponding to symbols in l_syms    
    ar_price = df_close.loc[date][l_syms].values  
    ar_value = ar_price * ar_shares  # array of actual dollar amount invested in each symbol
    portf_value = sum(ar_value)  # total actual portfolio value
    if verbose:
        print(f'{date = }, {l_syms = }, {ar_price = }, {ar_shares = }, {ar_value = }, {portf_value = }')            
        print(f'{date} {portf_value = }')
    return date, l_syms, ar_price, ar_shares, ar_value, portf_value  


def calc_portf_value_date_n(dates_in_days_lookbacks, my_symbols, df_close, my_portf_shares, my_SPY_shares, n, verbose=False):

  from ast import literal_eval
  from work11 import any_not_in_list, get_trading_date_n


 # Trading dates
  idx_close = df_close.index.strftime('%Y-%m-%d')

  # # Symbols in df_close
  # symbols_df_close = df_close.columns  # symbols in df_close   



  z_date_syms_shares = zip(dates_in_days_lookbacks, my_symbols, my_portf_shares, my_SPY_shares)

  date_exec = []  # buy date of portfolio
  shares_syms = []  # lists of shares of each symbol brought on date
  value_portf = [] # list of porfolio value on date
  shares_SPY = []  # list of shares of SPY brought on date
  value_SPY = []  # list of value of SPY shares on date 

  for date, symbols, portf_shares, SPY_shares in z_date_syms_shares:
    next_date_n = get_trading_date_n(date, idx_close, n, verbose=False)
    close_date_n = is_date_in_close(next_date_n, df_close)

    if close_date_n is None:
      p_date_exec = None
      p_ar_shares = None
      p_value_portf = None  # set to None when data are not available in df_close
      SPY_ar_shares = None
      SPY_value_portf = None # set to None when data are not available in df_close

      if verbose:
        print(f"No data for close_date_n {close_date_n}, pick's portf value = None")
        # print(f'No data for next_date_n {next_date_n}, SPY portf value =    None')

    else: 
      if portf_shares is None:
        p_date_exec = None
        p_ar_shares = None
        p_value_portf = None
        SPY_ar_shares = None
        SPY_value_portf = None

      else:  
        p_date_exec, p_ar_syms, p_ar_price, p_ar_shares, p_ar_value, p_value_portf = \
          calc_portf_value(df_close, close_date_n, symbols, portf_shares, verbose)
        
        SPY = str(['SPY'])
        SPY_date_exec, SPY_ar_syms, SPY_ar_price, SPY_ar_shares, SPY_ar_value, SPY_value_portf = \
          calc_portf_value(df_close, close_date_n, SPY, SPY_shares, verbose) 

        if verbose:
          print(f"next_date_n pick's portf value = {p_value_portf}")
          print(f'next_date_n SPY portf value =    {SPY_value_portf}')

    date_exec.append(p_date_exec)
    shares_syms.append(p_ar_shares)
    value_portf.append(p_value_portf)
    shares_SPY.append(SPY_ar_shares)
    value_SPY.append(SPY_value_portf)

    print('='*20, '\n')

  return date_exec, shares_syms, value_portf, shares_SPY, value_SPY  


###############################
def calc_portfolio_on_date(df_close, date, symbols, portf_shares=None, verbose=False):
  """
  Calculates the portfolio value and shares for a given date and list of symbols.

  Args:
    df_close: A DataFrame containing closing prices for the symbols.
    date: The date for which to calculate the portfolio.
    symbols: A list of symbols to buy or update.
    portf_shares: (Optional) A list of pre-existing portfolio shares. Used by `calc_portf_value_date_n`.
    verbose: (Optional) A boolean flag to control the amount of printed information.

  Returns:
    date_exec: The trading date on which the portfolio was bought or updated.
    symbols: The list of symbols in the portfolio.
    ar_shares: A list of number of shares for each symbol.
    ar_price: A list of closing prices for each symbol.
    ar_value: A list of the value of each position.
    portf_value: The total value of the portfolio.
  """

  # Calculate the next trading date if needed
  if portf_shares is None:
    ar_shares, ar_price, ar_value, portf_value = None, None, None, None
  else:
    next_date = get_trading_date_n(date, df_close.index, n=1, verbose=verbose)
    close_date_n = is_date_in_close(next_date, df_close)
    
    if close_date_n is None:
      if verbose:
        print(f"No data for close_date_n {close_date_n}, pick's portf value = None")
      ar_shares, ar_price, ar_value, portf_value = None, None, None, None
    else:
      ar_shares, ar_price, ar_value, portf_value = calc_portf_value(df_close, close_date_n, symbols, portf_shares, verbose=verbose)

  # Update date and symbol information
  date_exec = close_date_n if close_date_n else date
  symbols = str(['SPY']) if symbols is None else symbols

  return date_exec, symbols, ar_shares, ar_price, ar_value, portf_value



###############################