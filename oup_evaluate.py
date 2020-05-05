# Module imports
import math
import argparse
import numpy as np
from scipy.stats import norm
from alpha_vantage.foreignexchange import ForeignExchange

''' Define a function that fits OUP using linear (least-squares regression) '''
def lin_fit_OUP(price_history, T=1, steps=200):
    
    # Estimate the parameters using the methods used in the literature
    n = len(price_history) - 1
    np_stock = np.array(price_history)
    
    Sx = sum(price_history[:-1])
    Sy = sum(price_history[1:])
    Sxx = np.sum(np_stock[:-1] ** 2)
    Sxy = np.sum(np_stock[:-1] * np_stock[1:])
    Syy = np.sum(np_stock[1:] ** 2)

    a = ((n * Sxy) - (Sx * Sy)) / ((n* Sxx) - (Sx ** 2))
    b = (Sy - (a * Sx)) / n
    sde = math.sqrt(((n * Syy) - (Sy ** 2) - (a * ((n * Sxy) - (Sx * Sy)))) / (n * (n - 2)))

    # dX = alpha*(mu-X)*dt + sigma*dW
    delta = T / steps

    # alpha = - ln a / delta
    alpha = - (math.log(a) / delta)

    # mu = b / 1 - a
    mu = b / (1 - a)

    # sigma = sd(epsilon) * sqrt( -2 ln a / delta * (1 - a^2) )
    sigma = sde * math.sqrt((-2 * math.log(a)) / (delta * (1 - a ** 2)))  
    return alpha, mu, sigma

''' Trade OUP with selected parameters - real-time '''
def trade_OUP(stock_price, trading_time=70, ma_period=21, ma_t="ema", T=1, quantile=0.25):
    
    # Ensure the parameters are valid
    s_len = len(stock_price)
    assert trading_time + ma_period < s_len, "Trading time with moving average period must be smaller than series length."

    # Keep track of lines
    top_band = []
    bot_band = []
    ma = []
    
    # Current state of trading
    holding = False
    buy_point = None
    
    # If the moving average is ema, we need an initial value
    if ma_t == "ema":
        ema = stock_price[s_len - trading_time - ma_period]
        for index in range(s_len - trading_time - ma_period + 1, s_len - trading_time):
            ema = (stock_price[index] - ema) * (2 / (ma_period + 1)) + ema
        ma.append(ema)
    
    # Go through the series and trade OUP
    for point_index in range(s_len - trading_time, s_len):
        
        # Identify the indexes for the start and end of the relevant series
        start = point_index - ma_period + 1
        end = point_index + 1
        
        # Exponential moving average
        if ma_t == "ema":
            ema = (stock_price[point_index] - ma[-1]) * (2 / (ma_period + 1)) + ma[-1]
            ma.append(ema)
        
        # Simple moving average 
        else:
            sma = sum(stock_price[start:end]) / ma_period
            ma.append(sma) 
        
        # Define the rest of the corridor
        try:
            
            # We are looking for the probability distribution at point 'point_index', using the series 
            # of len ma_period before it
            alpha, mu, sigma = lin_fit_OUP(stock_price[start:end], steps=ma_period, T=T)

            # Calculate the probability distribution at this point
            expected = mu + ((stock_price[start] - mu) * math.exp(-alpha * T))
            standard_dev = sigma * math.sqrt((1 - math.exp(-2 * alpha * T)) / (2 * alpha))
            quantiles = norm.ppf([quantile, 0.5, 1 - quantile], expected, standard_dev)
            
            # Calculate the offsets
            top = ma[-1] + (quantiles[2] - quantiles[1])
            bot = ma[-1] - (quantiles[1] - quantiles[0])

        # In the event we cannot perform the fit with least-squares
        except:
            
            # Use the previous offset if we can
            try:
                offset_top = top_band[-1] - ma[-2]
                top = ma[-1] + offset_top
                
                offset_bot = ma[-2] - bot_band[-1]
                bot = ma[-1] - offset_bot
                
            # Otherwise, use offset = 5% of price
            except:
                offset = 0.05 * stock_price[point_index]
                top = ma[-1] + offset
                bot = ma[-1] - offset
        
        # Append the top and bottom regardless of how we generate them
        top_band.append(top)
        bot_band.append(bot)
        
        # Figure out if we need to buy or sell
        if stock_price[point_index] <= bot and not holding:
            holding = True 
            buy_point = s_len - point_index
        elif stock_price[point_index] >= top and holding:
            holding = False
    return holding, buy_point

# If the strategy is run, parse the command line arguments and execute the algorithm
if __name__ == "__main__":

	# -------------- PARSE COMMAND LINE ARGUMENTS --------------

	parser = argparse.ArgumentParser(description="Execute the OUP strategy on a current pair")

	# Required for the strategy to work, we need an asset to trade, specified as a currency pair
	parser.add_argument("from", help="symbol of currency exchanged from, i.e. the pair formed is FROM/TO")
	parser.add_argument("to", help="symbol of currency exhchanged to, i.e. the pair formed is FROM/TO ")

	# All additional arguments are those that can be specified as part of the OUP strategy
	parser.add_argument("-t", "--time", help="number of days to trade the asset for up to current day, defaults to 70", type=int, default=70)
	parser.add_argument("-p", "--period", help="moving average period used, defaults to 21", type=int, default=21)
	parser.add_argument("-m", "--ma_t", help="moving average type to use, options are \"ema\" and \"sma\", defaults to \"ema\"", default="ema")
	parser.add_argument("-T", "--horizon", help="time horizon to use when fitting OUP to series, defaults to 1", type=int, default=1)
	parser.add_argument("-q", "--quantile", help="quantile used, defaults to 0.25", type=float, default=0.25)
	args = parser.parse_args()

	# List of available currencies as documented on:
	# https://www.alphavantage.co/documentation/
	currencies = ['AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN', 'BAM', 'BBD', 'BDT', 'BGN', 'BHD', 'BIF', 'BMD', \
	'BND', 'BOB', 'BRL', 'BSD', 'BTN', 'BWP', 'BZD', 'CAD', 'CDF', 'CHF', 'CLF', 'CLP', 'CNH', 'CNY', 'COP', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', \
	'DOP', 'DZD', 'EGP', 'ERN', 'ETB', 'EUR', 'FJD', 'FKP', 'GBP', 'GEL', 'GHS', 'GIP', 'GMD', 'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK', 'HTG', \
	'HUF', 'IDR', 'ILS', 'INR', 'IQD', 'IRR', 'ISK', 'JEP', 'JMD', 'JOD', 'JPY', 'KES', 'KGS', 'KHR', 'KMF', 'KPW', 'KRW', 'KWD', 'KYD', 'KZT', \
	'LAK', 'LBP', 'LKR', 'LRD', 'LSL', 'LYD', 'MAD', 'MDL', 'MGA', 'MKD', 'MMK', 'MNT', 'MOP', 'MRO', 'MRU', 'MUR', 'MVR', 'MWK', 'MXN', 'MYR', \
	'MZN', 'NAD', 'NGN', 'NOK', 'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR', 'PLN', 'PYG', 'QAR', 'RON', 'RSD', 'RUB', 'RUR', 'RWF', \
	'SAR', 'SBD', 'SCR', 'SDG', 'SEK', 'SGD', 'SHP', 'SLL', 'SOS', 'SRD', 'SYP', 'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP', 'TRY', 'TTD', 'TWD', \
	'TZS', 'UAH', 'UGX', 'USD', 'UYU', 'UZS', 'VND', 'VUV', 'WST', 'XAF', 'XAG', 'XAU', 'XCD', 'XDR', 'XOF', 'XPF', 'YER', 'ZAR', 'ZMW', 'ZWL']

	# -------------- VALIDATE ALL COMMAND LINE INPUTS --------------

	# Validate currency codes
	from_ = getattr(args, "from")
	to_ = getattr(args, "to")
	assert from_ in currencies and to_ in currencies, "Currency codes must be supported by the Alpha Vantage API. See https://www.alphavantage.co/documentation/ for details."

	# Validate trading time param
	time_ = getattr(args, "time")
	assert 1 <= time_  and time_ < 100, "Trading time must be 1 <= TIME < 100 to be valid. Upper limit defined by API limitations."

	# Validate moving average period
	period_ = getattr(args, "period")
	assert 1 <= period_ and period_ < 100, "Moving average period must be 1 <= PERIOD < 100 to be valid. Upper limit defined by API limitations."

	# Validate moving average type
	mas = ["sma", "ema"]
	ma_t_ = getattr(args, "ma_t")
	assert ma_t_ in mas, "Moving average type must be one of \"ema\" or \"sma\"."

	# Validate time horizon
	t_ = getattr(args, "horizon")
	assert 1 <= t_ and t_ < 100, "Time horizon must be greater than or equal to 1 and less than the size of the series for fitting."

	# Validate quantile
	q_ = getattr(args, "quantile")
	assert 0 < q_ and q_ < 0.5, "Quantile sample must be in the range 0 < QUANTILE < 0.5 to function."

	# --------- USE THE OUP STRATEGY TO MAKE A BUY/SELL DECISION ---------

	# Process of retrieving can take a few seconds, inform the user
	print("\tRetrieving past price values, please wait...\n")

	# Retrieve the price history (past 100 days) for the given currency pair
	# DISCLAIMER: Alpha Vantage API puts a limit on calls to 5 per minute and 500 per day
	# 			  Ideally, the API key would also be unique per user, but is included for demonstration purposes
	try:
		cc = ForeignExchange(key='UH8X7NHOQEPGPWUV', output_format='pandas')
		data, meta_data = cc.get_currency_exchange_daily(from_symbol=from_, to_symbol=to_)


		# Execute the strategy with the given parameters
		bought, last_buy_point = trade_OUP(data['4. close'].tolist(), trading_time=time_, ma_period=period_, ma_t=ma_t_, T=t_, quantile=q_)
		pair = str(from_) + "/" + str(to_)

		# Provide command line feedback
		if bought:
			print("\tOUP strategy would purchase " + pair + ". Buy order set " + str(last_buy_point) + " day(s) ago.")
		else:
			print("\tOUP strategy would sell " + pair + ".")

	# Handle the errors the API can return
	except ValueError as e:
		if "Thank you" in str(e):
			print("\tLimit on API calls reached. Please try again later.")
		else:
			print("\tAPI does not support this pair combination. Please use a different currency pair.")