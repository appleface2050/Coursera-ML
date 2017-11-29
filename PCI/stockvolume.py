import urllib

tickers = ['YHOO', 'AVP', 'BIIB', 'BP', 'CL', 'CVX',
           'DNA', 'EXPE', 'GOOG', 'PG', 'XOM', 'AMGN']

shortest = 300
prices = {}
dates = None

for t in tickers:

    url = 'http://ichart.finance.yahoo.com/table.csv?' + \
          's=%s&d=11&e=26&f=2006&g=d&a=3&b=12&c=1996' % t + \
          '&ignore=.csv'
    print(url)
    rows = urllib.request.urlopen(url).readlines()
    prices[t] = [float(r.split(',')[5]) for r in rows[1:] if r.strip() != '']
    if len(prices[t]) < shortest: shortest = len(prices[t])

    if not dates:
        dates = [r.split(',')[0] for r in rows[1:] if r.strip() != '']

pass
