"""MC1-P2: Optimize a portfolio."""
import sys
sys.path.append('../')


import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
import math

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


def sharp_ratio_fun(allocs, prices):
    port_val = get_portfolio_value(prices, allocs, 1000000)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    return sharpe_ratio * -1

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normed = prices_SPY / prices_SPY.ix[0] # normalize prices for SPY


    # find the allocations for the optimal portfolio

    ### FIND THE ALLOCATIONS
    num_syms = len(syms)
    guess_allocs = [1.0/num_syms for x in syms]
#    print guess_allocs

    cons = ({ 'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)})

    bounds_for_optimization = tuple([(0.0, 1.0) for x in syms])
#    print bounds_for_optimization

    min_result = spo.minimize(sharp_ratio_fun, guess_allocs, args=(prices,), bounds= bounds_for_optimization , constraints=cons, method='SLSQP')
    allocs = min_result.x
    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0

    ### COMPUTE STATS
    # Get daily portfolio value (already normalized since we use default start_val=1.0)
    port_val = get_portfolio_value(prices, allocs)


    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = get_portfolio_stats(port_val)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY_normed], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily Portfolio Value and SPY", save_plot = "./report.png")

    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']


    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,  syms = symbols,  gen_plot = True)

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
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
