# print(list_of_lists)
symbols_picks = [val for sublist in list_of_lists for val in sublist]
print(f'symbol count from model picks: {len(symbols_picks)}')
# list sorted unique symbols
unique_symbols_picks = sorted(list(set(symbols_picks)))
print(f'unique symbol count from model picks: {len(unique_symbols_picks)}')
# print(len(unique_symbols_picks))
# unique_symbols_picks
