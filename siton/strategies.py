"""Strategy registry — one line per strategy using indicator wrappers.

To add a new strategy, append to ALL_STRATEGIES using the appropriate
factory from siton.indicators. See indicators.py docstring for the full
list of available factory patterns.
"""

import talib
from siton.composites import COMPOSITE_STRATEGIES
from siton.indicators import (
    # Phase 1 — existing
    crossover, threshold, zero_cross, dual_zero_cross,
    macd, stochrsi, bollinger,
    # Phase 2 — close-only
    price_crossover, mama_crossover, ratio_cross, macdext, macdfix,
    cycle_crossover, phasor_crossover, ht_trendmode, breakout,
    # Phase 3 — HL / HLC / HLCV
    zero_cross_hl, aroon_crossover, price_crossover_hl,
    sar_crossover, sarext_crossover, dm_crossover,
    threshold_hlc, threshold_hlcv, trend_strength,
    ultosc_threshold, stoch_hlc, stochf_hlc,
    di_crossover, volatility_breakout,
    # Phase 4 — OHLC / Volume
    candlestick, bop_signal, price_transform_crossover,
    obv_crossover, ad_crossover, adosc_zero_cross,
)

ALL_STRATEGIES = [

    # ===================================================================
    # PHASE 1 — Existing strategies (15)
    # ===================================================================

    # ---- Crossover family ----
    crossover("SMA_Cross",  talib.SMA,  fast=[3, 5, 8, 10, 13, 15, 20, 25], slow=[30, 40, 50, 75, 100, 150, 200]),
    crossover("EMA_Cross",  talib.EMA,  fast=[3, 5, 8, 9, 12, 15, 20, 25],  slow=[21, 26, 30, 40, 50, 75, 100]),
    crossover("DEMA_Cross", talib.DEMA, fast=[3, 5, 8, 10, 15, 20, 25],     slow=[30, 40, 50, 75, 100, 150, 200]),
    crossover("TEMA_Cross", talib.TEMA, fast=[3, 5, 8, 10, 15, 20, 25],     slow=[30, 40, 50, 75, 100, 150, 200]),
    crossover("WMA_Cross",  talib.WMA,  fast=[3, 5, 8, 10, 15, 20, 25],     slow=[30, 40, 50, 75, 100, 150, 200]),
    crossover("KAMA_Cross", talib.KAMA, fast=[5, 8, 10, 15, 20],            slow=[25, 30, 40, 50, 75, 100]),

    # ---- Oscillators (mean reversion) ----
    threshold("RSI", talib.RSI, periods=[5, 7, 10, 14, 21, 28],
              oversold=[15, 20, 25, 30, 35], overbought=[65, 70, 75, 80, 85]),
    threshold("CMO", talib.CMO, periods=[5, 7, 10, 14, 21, 28],
              oversold=[-60, -50, -40, -30], overbought=[30, 40, 50, 60]),
    stochrsi("StochRSI", periods=[7, 14, 21], fastk=[3, 5, 7], fastd=[3, 5],
             oversold=[15, 20, 25, 30], overbought=[70, 75, 80, 85]),

    # ---- MACD family ----
    macd("MACD", fast=[5, 6, 8, 10, 12, 15], slow=[18, 21, 24, 26, 30], signal_period=[5, 7, 9, 12]),
    dual_zero_cross("APO", talib.APO, fast=[5, 8, 10, 12], slow=[20, 26, 30, 40]),
    dual_zero_cross("PPO", talib.PPO, fast=[5, 8, 10, 12], slow=[20, 26, 30, 40]),

    # ---- Momentum ----
    zero_cross("MOM", talib.MOM, periods=[3, 5, 7, 10, 14, 21, 28]),
    zero_cross("ROC", talib.ROC, periods=[3, 5, 7, 10, 14, 21, 28]),

    # ---- Volatility ----
    bollinger("Bollinger", windows=[10, 14, 20, 25, 30, 40], num_std=[1.0, 1.5, 2.0, 2.5, 3.0]),

    # ===================================================================
    # PHASE 2 — New close-only strategies (19)
    # ===================================================================

    # ---- Crossover (new MA type) ----
    crossover("TRIMA_Cross", talib.TRIMA, fast=[5, 8, 10, 15, 20], slow=[30, 40, 50, 75, 100]),

    # ---- Zero-cross momentum ----
    zero_cross("TRIX", talib.TRIX, periods=[5, 10, 14, 20, 30]),
    zero_cross("ROCP", talib.ROCP, periods=[3, 5, 7, 10, 14, 21, 28]),
    zero_cross("LR_Slope", talib.LINEARREG_SLOPE, periods=[7, 10, 14, 20, 30]),
    zero_cross("LR_Angle", talib.LINEARREG_ANGLE, periods=[7, 10, 14, 20, 30]),

    # ---- Ratio cross ----
    ratio_cross("ROCR", talib.ROCR, periods=[3, 5, 7, 10, 14, 21, 28], center=1.0),
    ratio_cross("ROCR100", talib.ROCR100, periods=[3, 5, 7, 10, 14, 21, 28], center=100.0),

    # ---- MACD variants ----
    macdext("MACDEXT", fast=[8, 12], slow=[21, 26], signal_period=[7, 9], ma_types=[0, 1, 2, 3, 4, 8]),
    macdfix("MACDFIX", signal_periods=[5, 7, 9, 12, 15]),

    # ---- MAMA ----
    mama_crossover("MAMA", fast_limits=[0.25, 0.5, 0.75], slow_limits=[0.01, 0.03, 0.05, 0.1]),

    # ---- Price crossover (close vs overlay) ----
    price_crossover("MIDPOINT", talib.MIDPOINT, periods=[7, 10, 14, 20, 30, 50]),
    price_crossover("LINEARREG", talib.LINEARREG, periods=[7, 10, 14, 20, 30]),
    price_crossover("TSF", talib.TSF, periods=[7, 10, 14, 20, 30]),
    price_crossover("HT_Trendline", talib.HT_TRENDLINE, periods=None),

    # ---- Hilbert Transform ----
    cycle_crossover("HT_Sine"),
    phasor_crossover("HT_Phasor"),
    ht_trendmode("HT_TrendMode"),

    # ---- Channel breakout ----
    breakout("Chan_Breakout", periods=[10, 14, 20, 30, 50]),

    # ===================================================================
    # PHASE 3 — HL / HLC / HLCV strategies (18)
    # ===================================================================

    # ---- Aroon ----
    aroon_crossover("AROON", periods=[7, 10, 14, 20, 25, 30]),
    zero_cross_hl("AROONOSC", talib.AROONOSC, periods=[7, 10, 14, 20, 25, 30]),

    # ---- MIDPRICE (HL) ----
    price_crossover_hl("MIDPRICE", talib.MIDPRICE, periods=[7, 10, 14, 20, 30, 50]),

    # ---- SAR ----
    sar_crossover("SAR",
                  accelerations=[0.01, 0.02, 0.03, 0.04],
                  maximums=[0.1, 0.2, 0.3, 0.4]),
    sarext_crossover("SAREXT",
                     accel_inits=[0.01, 0.02],
                     accelerations=[0.02, 0.03],
                     maximums=[0.1, 0.2, 0.3]),

    # ---- Directional Movement ----
    dm_crossover("DM_Cross", periods=[7, 10, 14, 20, 28]),
    di_crossover("DI_Cross", periods=[7, 10, 14, 20, 28]),

    # ---- Trend strength ----
    trend_strength("ADX", talib.ADX, periods=[7, 10, 14, 20, 28], thresholds=[20, 25, 30, 35, 40]),
    trend_strength("ADXR", talib.ADXR, periods=[7, 10, 14, 20, 28], thresholds=[20, 25, 30, 35, 40]),
    trend_strength("DX", talib.DX, periods=[7, 10, 14, 20, 28], thresholds=[20, 25, 30, 35, 40]),

    # ---- HLC oscillators ----
    threshold_hlc("CCI", talib.CCI, periods=[7, 10, 14, 20, 28],
                  oversold=[-200, -150, -100], overbought=[100, 150, 200]),
    threshold_hlc("WILLR", talib.WILLR, periods=[7, 10, 14, 21, 28],
                  oversold=[-90, -80, -70], overbought=[-30, -20, -10]),

    # ---- HLCV oscillator ----
    threshold_hlcv("MFI", talib.MFI, periods=[7, 10, 14, 21, 28],
                   oversold=[15, 20, 25, 30], overbought=[70, 75, 80, 85]),

    # ---- Ultimate Oscillator ----
    ultosc_threshold("ULTOSC", p1_list=[5, 7], p2_list=[10, 14], p3_list=[21, 28],
                     oversold=[25, 30], overbought=[70, 75]),

    # ---- Stochastic ----
    stoch_hlc("STOCH", fastk_periods=[5, 9, 14], slowk_periods=[3, 5], slowd_periods=[3, 5],
              oversold=[15, 20, 25], overbought=[75, 80, 85]),
    stochf_hlc("STOCHF", fastk_periods=[5, 9, 14], fastd_periods=[3, 5],
               oversold=[15, 20, 25], overbought=[75, 80, 85]),

    # ---- Volatility breakout ----
    volatility_breakout("ATR_Brkout", talib.ATR, periods=[7, 10, 14, 20], lookbacks=[14, 20, 30]),
    volatility_breakout("NATR_Brkout", talib.NATR, periods=[7, 10, 14, 20], lookbacks=[14, 20, 30]),

    # ===================================================================
    # PHASE 4 — OHLC / OHLCV / Volume strategies (66+)
    # ===================================================================

    # ---- Balance of Power ----
    bop_signal("BOP"),

    # ---- Price transforms ----
    price_transform_crossover("AVGPRICE", talib.AVGPRICE, ["open", "high", "low", "close"]),
    price_transform_crossover("TYPPRICE", talib.TYPPRICE, ["high", "low", "close"]),
    price_transform_crossover("WCLPRICE", talib.WCLPRICE, ["high", "low", "close"]),
    price_transform_crossover("MEDPRICE", talib.MEDPRICE, ["high", "low"]),

    # ---- Volume indicators ----
    obv_crossover("OBV", ma_periods=[10, 20, 30, 50, 100]),
    ad_crossover("AD", ma_periods=[10, 20, 30, 50, 100]),
    adosc_zero_cross("ADOSC", fast_periods=[3, 5, 8, 10], slow_periods=[10, 20, 26, 40]),

    # ---- Candlestick patterns (no penetration) — 54 patterns ----
    candlestick("CDL2CROWS", talib.CDL2CROWS),
    candlestick("CDL3BLACKCROWS", talib.CDL3BLACKCROWS),
    candlestick("CDL3INSIDE", talib.CDL3INSIDE),
    candlestick("CDL3LINESTRIKE", talib.CDL3LINESTRIKE),
    candlestick("CDL3OUTSIDE", talib.CDL3OUTSIDE),
    candlestick("CDL3STARSINSOUTH", talib.CDL3STARSINSOUTH),
    candlestick("CDL3WHITESOLDIERS", talib.CDL3WHITESOLDIERS),
    candlestick("CDLADVANCEBLOCK", talib.CDLADVANCEBLOCK),
    candlestick("CDLBELTHOLD", talib.CDLBELTHOLD),
    candlestick("CDLBREAKAWAY", talib.CDLBREAKAWAY),
    candlestick("CDLCLOSINGMARUBOZU", talib.CDLCLOSINGMARUBOZU),
    candlestick("CDLCONCEALBABYSWALL", talib.CDLCONCEALBABYSWALL),
    candlestick("CDLCOUNTERATTACK", talib.CDLCOUNTERATTACK),
    candlestick("CDLDOJI", talib.CDLDOJI),
    candlestick("CDLDOJISTAR", talib.CDLDOJISTAR),
    candlestick("CDLDRAGONFLYDOJI", talib.CDLDRAGONFLYDOJI),
    candlestick("CDLENGULFING", talib.CDLENGULFING),
    candlestick("CDLGAPSIDESIDEWHITE", talib.CDLGAPSIDESIDEWHITE),
    candlestick("CDLGRAVESTONEDOJI", talib.CDLGRAVESTONEDOJI),
    candlestick("CDLHAMMER", talib.CDLHAMMER),
    candlestick("CDLHANGINGMAN", talib.CDLHANGINGMAN),
    candlestick("CDLHARAMI", talib.CDLHARAMI),
    candlestick("CDLHARAMICROSS", talib.CDLHARAMICROSS),
    candlestick("CDLHIGHWAVE", talib.CDLHIGHWAVE),
    candlestick("CDLHIKKAKE", talib.CDLHIKKAKE),
    candlestick("CDLHIKKAKEMOD", talib.CDLHIKKAKEMOD),
    candlestick("CDLHOMINGPIGEON", talib.CDLHOMINGPIGEON),
    candlestick("CDLIDENTICAL3CROWS", talib.CDLIDENTICAL3CROWS),
    candlestick("CDLINNECK", talib.CDLINNECK),
    candlestick("CDLINVERTEDHAMMER", talib.CDLINVERTEDHAMMER),
    candlestick("CDLKICKING", talib.CDLKICKING),
    candlestick("CDLKICKINGBYLENGTH", talib.CDLKICKINGBYLENGTH),
    candlestick("CDLLADDERBOTTOM", talib.CDLLADDERBOTTOM),
    candlestick("CDLLONGLEGGEDDOJI", talib.CDLLONGLEGGEDDOJI),
    candlestick("CDLLONGLINE", talib.CDLLONGLINE),
    candlestick("CDLMARUBOZU", talib.CDLMARUBOZU),
    candlestick("CDLMATCHINGLOW", talib.CDLMATCHINGLOW),
    candlestick("CDLONNECK", talib.CDLONNECK),
    candlestick("CDLPIERCING", talib.CDLPIERCING),
    candlestick("CDLRICKSHAWMAN", talib.CDLRICKSHAWMAN),
    candlestick("CDLRISEFALL3METHODS", talib.CDLRISEFALL3METHODS),
    candlestick("CDLSEPARATINGLINES", talib.CDLSEPARATINGLINES),
    candlestick("CDLSHOOTINGSTAR", talib.CDLSHOOTINGSTAR),
    candlestick("CDLSHORTLINE", talib.CDLSHORTLINE),
    candlestick("CDLSPINNINGTOP", talib.CDLSPINNINGTOP),
    candlestick("CDLSTALLEDPATTERN", talib.CDLSTALLEDPATTERN),
    candlestick("CDLSTICKSANDWICH", talib.CDLSTICKSANDWICH),
    candlestick("CDLTAKURI", talib.CDLTAKURI),
    candlestick("CDLTASUKIGAP", talib.CDLTASUKIGAP),
    candlestick("CDLTHRUSTING", talib.CDLTHRUSTING),
    candlestick("CDLTRISTAR", talib.CDLTRISTAR),
    candlestick("CDLUNIQUE3RIVER", talib.CDLUNIQUE3RIVER),
    candlestick("CDLUPSIDEGAP2CROWS", talib.CDLUPSIDEGAP2CROWS),
    candlestick("CDLXSIDEGAP3METHODS", talib.CDLXSIDEGAP3METHODS),

    # ---- Candlestick patterns (with penetration) — 7 patterns x 4 values ----
    candlestick("CDLABANDONEDBABY", talib.CDLABANDONEDBABY, penetration_values=[0.1, 0.2, 0.3, 0.5]),
    candlestick("CDLDARKCLOUDCOVER", talib.CDLDARKCLOUDCOVER, penetration_values=[0.3, 0.4, 0.5, 0.6]),
    candlestick("CDLEVENINGDOJISTAR", talib.CDLEVENINGDOJISTAR, penetration_values=[0.1, 0.2, 0.3, 0.5]),
    candlestick("CDLEVENINGSTAR", talib.CDLEVENINGSTAR, penetration_values=[0.1, 0.2, 0.3, 0.5]),
    candlestick("CDLMATHOLD", talib.CDLMATHOLD, penetration_values=[0.3, 0.4, 0.5, 0.6]),
    candlestick("CDLMORNINGDOJISTAR", talib.CDLMORNINGDOJISTAR, penetration_values=[0.1, 0.2, 0.3, 0.5]),
    candlestick("CDLMORNINGSTAR", talib.CDLMORNINGSTAR, penetration_values=[0.1, 0.2, 0.3, 0.5]),
]

ALL_STRATEGIES += COMPOSITE_STRATEGIES
