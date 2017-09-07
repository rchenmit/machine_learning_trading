"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
import sys
sys.path.append('../')
from util import get_data, plot_data
import math


def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """


    normalized_prices = prices / prices.ix[0, :]
    allocated_amounts = normalized_prices * allocs
    pos_vals = allocated_amounts * start_val
    port_val = pos_vals.sum(axis=1)
#    print port_val

    return port_val

def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    #TODO: Your code here

    ldt_timestamps = list(df.index)  # timestamps from dataframe
    ls_symbols = list(df.columns.values) # symbols for the stocks
    
    df_normalized = df.copy()
    df_normalized[ls_symbols] = df_normalized[ls_symbols].apply(lambda x: (x - x[0]) / x[0] + 1)

    na_normalized_price = df_normalized.values

    # print na_normalized_price[0:10][:]
    # for i in range(len(na_normalized_price[0])):
    #     na_normalized_price[:][i] = (na_normalized_price[:][i] - na_normalized_price[0][i]) / na_normalized_price[0][i]

    # print na_normalized_price[0:10][:]

    plt.clf()
    plt.plot(ldt_timestamps, na_normalized_price)
    plt.legend(ls_symbols)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('comparison.png', format='png')

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    # TODO: Your code here



    df_daily_ret = port_val.copy()
    df_daily_ret[1:] = (port_val[1:] / port_val[:-1].values) - 1
    df_daily_ret = df_daily_ret[1:]

    # retun these
    cum_ret = (port_val[-1]/port_val[0]) - 1
    avg_daily_ret = df_daily_ret.mean()
    std_daily_ret = df_daily_ret.std()
    sharpe_ratio = math.sqrt(samples_per_year) * (avg_daily_ret - daily_rf) / std_daily_ret


    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    ##port_val = prices_SPY # add code here to compute daily portfolio values
    port_val = get_portfolio_value(prices, allocs, sv)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    ##cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    cr, adr, sddr, sr = get_portfolio_stats(port_val)


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp, title="Daily portfolio value and SPY")

    # Add code here to properly compute end value
    ##ev = sv
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()

## test data ################################################################################

# Start Date: 2009-01-01 00:00:00
# End Date: 2010-01-01 00:00:00
# Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
# Allocations: [0.2, 0.3, 0.4, 0.1]
# Sharpe Ratio: 2.79622139929
# Volatility (stdev of daily returns): 0.0119577327574
# Average Daily Return: 0.00210629951522
# Cumulative Return: 0.665786603962