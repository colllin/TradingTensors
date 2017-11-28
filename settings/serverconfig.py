TOKEN = "efc6f5dc1bf5aaad1bada056d30e149e-0a8e9979916dc1b3862416c0333723b5" #Oanda Token
ID = "colllin" #Oanda Account ID

#Compute Returns
# Now=t,  we only know current open and last close
# True -> (OPEN[t] - OPEN[t-1])/(OPEN[t])
# False -> (CLOSE[t-1] - CLOSE[t-2]) / CLOSE[t-2]
# Note: We won't know what today's close is until tomorrow!
RETURNS_BY_OPEN = True

#ENVIRONMENT
HISTORICAL_DATA_LENGTH = 500
TRAIN_SPLIT = 0.6

#INDICATORS
from talib import ATR, SMA, RSI, BBANDS
