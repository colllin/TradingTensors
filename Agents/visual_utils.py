import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHART_SIZE = (20,10)

def rewardPlot(record, best_models, type, top_n=3):

    arr = np.asarray(record)

    #Return the index based on top n
    top_10_episodes = [x[0] for x in best_models]
    top_10_index = np.array(top_10_episodes) -1

    top_n_episodes = top_10_episodes[:top_n]
    top_n_index = top_10_index[:top_n]
    top_n_rewards = arr[top_n_index]


    fig = plt.figure(figsize=CHART_SIZE)
    ax = fig.add_subplot(111)
    color = 'b-' if type=='Total' else 'r-'
    ax.plot(record, color)
    ax.set_title("%s Reward (Showing Top %s)"%(type,top_n), fontdict={'fontsize':20})
    ax.set_xlabel("Episodes")


    textString = "TOP {}: \n".format(top_n)
    for i, r in enumerate(top_n_rewards):

        epi= top_n_episodes[i]

        textString += "Episode {}: {} \n".format(epi, record[epi-1])

    ax.text(0.75, 0.5, textString, fontsize=10, verticalalignment='top',transform=ax.transAxes,
    bbox={'alpha':0.5, 'pad':10})

    plt.show()


def ohlcPlot(trades, ohlc, equity_curve):
    trades = pd.DataFrame([vars(f) for f in trades])

    buys = trades.loc[trades.type=='BUY', :]
    sells = trades.loc[trades.type=='SELL', :]
    print(buys.columns)
    print ("Summary Statistics (Test)\n")

    print ("Total Trades            | {}        (Buy){}       (Sell){} "\
        .format(trades.shape[0], buys.shape[0], sells.shape[0]))

    #Calculate Profit breakdown
    total_profit = trades.profit.sum()
    buy_profit = buys.profit.sum()
    sell_profit = sells.profit.sum()

    print ("Profit (in pips)        | %.2f   (Buy)%.2f   (Sell)%.2f"\
        %(total_profit, buy_profit, sell_profit))

    #Calculate Win Ratio
    total_percent = (trades.loc[trades.profit>0,'profit'].count()/ trades.shape[0]) * 100
    buy_percent = (buys.loc[buys.profit>0, 'profit'].count()/buys.shape[0]) * 100
    sell_percent = (sells.loc[sells.profit>0, 'profit'].count()/sells.shape[0]) * 100
    print ("Win Ratio               | %.2f%%    (Buy)%.2f%%   (Sell)%.2f %%"%(total_percent, buy_percent, sell_percent))

    #Duration
    duration = trades.duration.mean()
    print ("Average Trade Duration  | %.2f"%(duration))

    #make OHLC ohlc matplotlib friendly
    datetime_index = mdates.date2num(ohlc.index.to_pydatetime())

    proper_feed = list(zip(
        datetime_index,
        ohlc.Open.tolist(),
        ohlc.High.tolist(),
        ohlc.Low.tolist(),
        ohlc.Close.tolist()
        ))

    #actual PLotting
    fig, (ax, ax2) = plt.subplots(2,1, figsize=CHART_SIZE)

    ax.set_title('Action History', fontdict={'fontsize':20})

    all_days= mdates.DayLocator()
    ax.xaxis.set_major_locator(all_days)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    #Candlestick chart
    mf.candlestick_ohlc(
        ax,
        proper_feed,
        width=0.02,
        colorup='green',
        colordown='red'
    )
    #Buy indicator
    ax.plot(
        mdates.date2num([time for time in buys.entry_time]),
        [price-0.001 for price in buys.entry_price],
        'b^',
        alpha=1.0
    )
    #Sell indicator
    ax.plot(
        mdates.date2num([time for time in sells.entry_time]),
        [price+0.001 for price in sells.entry_price],
        'rv',
        alpha=1.0
    )
    #Secondary Plot
    ax2.set_title("Equity")
    ax2.plot(equity_curve)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    plt.show()
