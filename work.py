def get_Invested_Capital(dict, index_dict):
  """
  Get net operating profit after tax from yahoo query for a symbol.
  dict is nested dictionary from a query of all modules for a symbol.
  index_dict is the index for the nested dictionaries, where index_dict=0 is the latest statement.
  The maximum for index_dict is 3 since yahoo returns 4 of the most recent statements.
  """  

  key_chain_bal_stmt_endDate = [index_dict, 'balanceSheetHistory', 'balanceSheetStatements', 'endDate']
  bal_stmt_end_date = chained_get(d_sorted_modules, *key_chain_bal_stmt_endDate)

  key_chain_totalLiab = [index_dict, 'balanceSheetHistory', 'balanceSheetStatements', 'totalLiab']
  totalLiab = chained_get(d_sorted_modules, *key_chain_totalLiab)

  key_chain_totalStockholderEquity = [index_dict, 'balanceSheetHistory', 'balanceSheetStatements', 'totalStockholderEquity']
  totalStockholderEquity = chained_get(d_sorted_modules, *key_chain_totalStockholderEquity)

  key_chain_cash = [index_dict, 'balanceSheetHistory', 'balanceSheetStatements', 'cash']
  cash = chained_get(d_sorted_modules, *key_chain_cash)

  key_chain_shortTermInvestments = [index_dict, 'balanceSheetHistory', 'balanceSheetStatements', 'shortTermInvestments']
  shortTermInvestments = chained_get(d_sorted_modules, *key_chain_shortTermInvestments)

  # https://seekingalpha.com/article/4486058-return-on-invested-capital
  Invested_Capital = totalLiab + totalStockholderEquity - (cash + shortTermInvestments)
  return Invested_Capital, totalLiab, totalStockholderEquity, cash, shortTermInvestments, bal_stmt_end_date