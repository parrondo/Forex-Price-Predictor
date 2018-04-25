import pandas as pd
import numpy as np
import talib as ta

     


def getCustomData(df):
    open = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    df['TRENDMODE'] = ta.HT_TRENDMODE(close)
    df['RSI'] = ta.RSI(close,timeperiod=14)
    df['RSI2'] = ta.RSI(close,timeperiod=7)
    df['RSI3'] = ta.RSI(close,timeperiod=28)
    df['ADX1'] = ta.ADX(high,low,close,timeperiod=7)
    df['ADX2'] = ta.ADX(high,low,close,timeperiod=28)
    df['ADX'] = ta.ADX(high,low,close,timeperiod=14)
    df['SLOWK'], df['SLOWD'] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['FASTK'], df['FASTD'] = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['ULTOSC'] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = ta.WILLR(high, low, close, timeperiod=14) 
    #df['SMA']= ta.SMA(close,timeperiod=30)
    #df['MA'] = ta.MA(close,timeperiod=30,matype=0)


'''Momentum Indicators'''
def getMomentumIndicators(df):

    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']
    df['ADX'] = ta.ADX(high, low, close, timeperiod=14)
    df['SMA'] = ta.ADXR(high, low, close, timeperiod=14)
    df['APO'] = ta.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df['AROONDOWN'], df['AROOONUP'] = ta.AROON(high, low, timeperiod=14)
    df['AROONOSC'] = ta.AROONOSC(high, low, timeperiod=14)
    df['BOP'] = ta.BOP(open, high, low, close)
    df['CCI'] = ta.CCI(high, low, close, timeperiod=14)
    df['CMO'] = ta.CMO(close, timeperiod=14)
    df['DX'] = ta.DX(high, low, close, timeperiod=14)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = ta.MFI(high, low, close, volume, timeperiod=14)
    df['MINUS_DI'] = ta.MINUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DM']= ta.MINUS_DM(high, low, timeperiod=14)
    df['MOM'] = ta.MOM(close, timeperiod=10)
    df['PLUS_DM'] =ta.PLUS_DM(high, low, timeperiod=14)
    df['PPO'] = ta.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    df['ROC'] = ta.ROC(close, timeperiod=10)
    df['ROCP'] = ta.ROCP(close, timeperiod=10)
    df['ROCR'] = ta.ROCR(close, timeperiod=10)
    df['ROCR100'] = ta.ROCR100(close, timeperiod=10)
    df['RSI'] = ta.RSI(close, timeperiod=14)
    df['SLOWK'], df['SLOWD'] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['FASTK'], df['FASTD'] = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['FASTK2'], df['FASTD2'] = ta.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TRIX'] = ta.TRIX(close, timeperiod=30)
    df['ULTOSC'] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = ta.WILLR(high, low, close, timeperiod=14)

'''Overlap Study Functions'''
def getOverlapFunctions(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']
    df['UPPERBB'],df['MIDDLEBB'],df['LOWERBB'] = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['DEMA'] = ta.DEMA(close,timeperiod=30)
    df['EMA'] = ta.EMA(close, timeperiod=30)
    df['HT_TREND'] = ta.HT_TRENDLINE(close)
    df['KAMA'] = ta.KAMA(close, timeperiod=30)
    df['MA'] = ta.MA(close, timeperiod=30, matype=0)
    #df['MAMA'],df['FAMA'] = ta.MAMA(close, fastlimit=0, slowlimit=0)
    #df['MAVP'] = ta.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
    df['MIDPOINT'] = ta.MIDPOINT(close, timeperiod=14)
    df['MIDPRICE'] = ta.MIDPRICE(high, low, timeperiod=14)
    df['SAR'] = ta.SAR(high, low, acceleration=0, maximum=0)
    df['SAREXT'] = ta.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    df['SMA'] = ta.SMA(close, timeperiod=30)
    df['T3'] = ta.T3(close, timeperiod=5, vfactor=0)
    df['TEMA'] = ta.TEMA(close, timeperiod=30)
    df['TRIMA'] = ta.TRIMA(close, timeperiod=30)
    df['WMA'] = ta.WMA(close, timeperiod=30)

'''Start Pattern Indicators'''
def getPatternIndicators(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']
    '''
    df['2CROWS'] = ta.CDL2CROWS(open, high, low, close)
    df['3BLACKCROWS'] = ta.CDL3BLACKCROWS(open, high, low, close)
    df['3INSIDE'] = ta.CDL3INSIDE(open, high, low, close)
    df['3LINESTRIKE'] = ta.CDL3LINESTRIKE(open, high, low, close)
    df['3OUTSIDE'] = ta.CDL3OUTSIDE(open, high, low, close)
    df['3STARSOUTH'] = ta.CDL3STARSINSOUTH(open, high, low, close)
    df['3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(open, high, low, close)
    df['ABANDONEDBABY'] = ta.CDLABANDONEDBABY(open, high, low, close, penetration=0)
    df['ADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(open, high, low, close)
    df['BELTHOLD'] = ta.CDLBELTHOLD(open, high, low, close)
    df['BREAKAWAY'] = ta.CDLBREAKAWAY(open, high, low, close)
    '''
    group = ta.get_function_groups() 
    for i in group['Pattern Recognition']:
        print i
        method = getattr(ta,i)
        df[i] = method(open,high,low,close)   


''' Price Transform Functions'''
def getPriceTransforms(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']

    df['AVGPRICE'] = ta.AVGPRICE(open, high, low, close)
    df['MEDPRICE'] = ta.MEDPRICE(high, low)
    df['TYPPRICE'] = ta.TYPPRICE(high, low, close)
    df['WCLPRICE'] = ta.WCLPRICE(high, low, close)

'''Statistic Functions'''
def getStatFunctions(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']

    df['BETA'] = ta.BETA(high, low, timeperiod=5)
    df['CORREL'] = ta.CORREL(high, low, timeperiod=30)
    df['LINREG'] = ta.LINEARREG(close, timeperiod=14)
    df['LINREGANGLE'] = ta.LINEARREG_ANGLE(close, timeperiod=14)
    df['LINREGINTERCEPT'] = ta.LINEARREG_INTERCEPT(close, timeperiod=14)
    df['LINREGSLOPE'] = ta.LINEARREG_SLOPE(close, timeperiod=14)
    df['STDDEV'] = ta.STDDEV(close, timeperiod=5, nbdev=1)
    df['TSF'] = ta.TSF(close, timeperiod=14)
    df['VAR'] = ta.VAR(close, timeperiod=5, nbdev=1)

'''Volatility Indicators'''
def getVolatilityIndicators(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']

    df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
    df['NATR'] = ta.NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = ta.TRANGE(high, low, close)

'''Volume Indicators'''
def getVolumeIndicators(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']

    df['AD'] = ta.AD(high, low, close, volume)
    df['ADOSC'] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['OBV']= ta.OBV(close, volume)

'''Cycle Indicators'''
def getCycleIndicators(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    open = df['Open']
    volume = df['Volume']

    df['DCPERIOD'] = ta.HT_DCPERIOD(close)
    df['DCPHASE'] = ta.HT_DCPHASE(close)
    df['INPHASE'],df['QUADRATURE'] = ta.HT_PHASOR(close)
    df['SINE'], df['LEADSINE'] = ta.HT_SINE(close)
    df['TRENDMODE'] = ta.HT_TRENDMODE(close)
    

def getAllIndicators(df):
    getMomentumIndicators(df)
    getOverlapFunctions(df)
    getPatternIndicators(df)
    getPriceTransforms(df)
    getStatFunctions(df)
    getVolatilityIndicators(df)
    getVolumeIndicators(df)
    getCycleIndicators(df)
    

