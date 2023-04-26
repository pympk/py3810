_symbols = ['NVDA', 'SPY', 'BTC-USD']
_types = ['EQUITY', 'ETF', 'CRYPTOCURRENCY']

z_syms_types = zip(_symbols, _types)
for _sym, _typ in z_syms_types:
  yq_sym = Ticker(_sym).all_modules  # modules for a symbol
  l_sym_modules = list(yq_sym.values())  # list with one nested-dict, keys are the modules 
  d_nested_modules = l_sym_modules[0]  # nested dict, keys are the modules  

  if _typ == 'EQUITY':
    keys_equity = list(d_nested_modules.keys())    
  if _typ == 'ETF':
    keys_ETF = list(d_nested_modules.keys())
  if _typ == 'CRYPTOCURRENCY':
    keys_cryto = list(d_nested_modules.keys())                                   
  
  print(f'modules in {_typ}')
  pprint(l_sym_modules, depth=3, indent=1)
  print('='*120, '\n'*2)
