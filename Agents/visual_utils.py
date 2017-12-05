import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHART_SIZE = (20,10)

def rewardPlot(record, best_models, type, top_n=3):
    fig = plt.figure(figsize=CHART_SIZE)
    ax = fig.add_subplot(111)
    color = 'b-' if type=='Total' else 'r-'
    ax.plot(record, color)
    ax.set_title("Rewards ", fontdict={'fontsize':20})
    ax.set_title("%s Rewards"%(type), fontdict={'fontsize':20})
    ax.set_xlabel("Episodes")

    arr = np.asarray(record)
    #Return the index based on top n
    top_10_episodes = [x[0] for x in best_models]
    top_10_index = np.array(top_10_episodes) -1

    top_n_episodes = top_10_episodes[:top_n]
    top_n_index = top_10_index[:top_n]
    top_n_rewards = arr[top_n_index]

    textString = "TOP {}: \n".format(top_n)
    for i, r in enumerate(top_n_rewards):
        epi= top_n_episodes[i]
        textString += "Episode {}: {} \n".format(epi, record[epi-1])
    ax.text(0.75, 0.5, textString, fontsize=10, verticalalignment='top',transform=ax.transAxes,
            bbox={'alpha':0.5, 'pad':10})
    plt.show()

def ohlcPlot(trades, ohlc, equity_curve):
    trades = pd.DataFrame(trades)

    # "right" trade for instrument 'EUR_USD' means exchanging from EUR to USD.
    lefts = trades.loc[trades.type=='LEFT', :]
    rights = trades.loc[trades.type=='RIGHT', :]
    print ("Summary Statistics (Test)\n")

    print ("Total Trades            | {}        (Buy Left){}       (Buy Right){} "\
        .format(trades.shape[0], lefts.shape[0], rights.shape[0]))

    #Calculate Profit breakdown
    total_profit = trades.profit.sum()
    left_profit = lefts.profit.sum()
    right_profit = rights.profit.sum()

    print ("Profit (return %%)        | %.4f   (Hold Left)%.4f   (Hold Right)%.4f"\
        %(total_profit, left_profit, right_profit))

    # Profit Opportunity
    profit_opportunity = np.sum([abs(ohlc.Open[i]-ohlc.Open[i-1]) for i in range(1,len(ohlc)-1)])
    # optimal_left_trades = [(ohlc.Open[i]-ohlc.Open[i-1]) for i in range(1,len(ohlc)-1)]
    print ("Maximum Profit Opportunity  | %.4f"%(profit_opportunity))

    #Calculate Win Ratio
    total_percent = (trades.loc[trades.profit>0,'profit'].count()/ trades.shape[0]) * 100
    left_percent = (lefts.loc[lefts.profit>0, 'profit'].count()/lefts.shape[0]) * 100
    right_percent = (rights.loc[rights.profit>0, 'profit'].count()/rights.shape[0]) * 100
    print ("Win Ratio               | %.2f%%    (Buy)%.2f%%   (Sell)%.2f %%"%(total_percent, left_percent, right_percent))

    #Duration
    duration = trades.duration.mean()
    print ("Average Trade Duration  | %.2f"%(duration))

    # #make OHLC ohlc matplotlib friendly
    # datetime_index = mdates.date2num(ohlc.index.to_pydatetime())
    #
    # proper_feed = list(zip(
    #     datetime_index,
    #     ohlc.Open.tolist(),
    #     ohlc.High.tolist(),
    #     ohlc.Low.tolist(),
    #     ohlc.Close.tolist()
    #     ))
    #
    # #actual PLotting
    # fig, (ax, ax2) = plt.subplots(2,1, figsize=CHART_SIZE)
    #
    # ax.set_title('Action History', fontdict={'fontsize':20})
    #
    # all_days= mdates.DayLocator()
    # ax.xaxis.set_major_locator(all_days)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    #
    # #Candlestick chart
    # mf.candlestick_ohlc(
    #     ax,
    #     proper_feed,
    #     width=0.02,
    #     colorup='green',
    #     colordown='red'
    # )
    # #Buy indicator
    # ax.plot(
    #     mdates.date2num([time for time in lefts.entry_time]),
    #     [price-0.001 for price in lefts.entry_price],
    #     'b^',
    #     alpha=1.0
    # )
    # #Sell indicator
    # ax.plot(
    #     mdates.date2num([time for time in rights.entry_time]),
    #     [price+0.001 for price in rights.entry_price],
    #     'rv',
    #     alpha=1.0
    # )
    # #Secondary Plot
    # ax2.set_title("Equity")
    # # ax2.plot(equity_curve)
    #
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(45)
    #
    # plt.show()
