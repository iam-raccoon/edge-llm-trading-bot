"""
1시간봉 추세추종 전략용 지표.

백테스트로 검증된 파라미터 (하락장에서도 +25%, 손익비 2.0):
  - 진입: 직전 20봉(1h) 고점 상향 돌파 (Donchian breakout) + 종가 > ma50(1h)
  - 국면: BTC 일봉 종가 > BTC 일봉 **ma20** 일 때만 신규 매수 (위험선호 구간)
          ma50에서 ma20으로 교체(2026-08-11). MA50은 후행이라 하락 초반을
          통과시킨다. 3.1년 재측정에서 ma20이 누적 +71%·최대낙폭 -41%.
  - 청산: 샹들리에 트레일링 = 진입후 최고가 - 3×ATR(14)
"""

import pyupbit

DC_ENTRY = 20       # 돌파 기준 봉수
ATR_N = 14
MA_TREND = 50       # 1h 장기추세 필터
MOM_LOOKBACK = 24   # 24시간 모멘텀 (후보 정렬용)
REGIME_MA = 20      # BTC 일봉 국면 판단 MA. 50 -> 20 (2026-08-11)
                    # 근거: 업비트 1시간봉 3.1년 재측정(htf_tune.py regime).
                    #   ma50 +218,884원 / 낙폭 -100,758 / 2025+ +30,523
                    #   ma20 +373,791원 / 낙폭  -59,549 / 2025+ +109,439
                    # 허용일 비율은 51%로 동일 — 더 자주 막는 게 아니라
                    # 더 이른 타이밍에 막는다. 필터 자체의 값어치도 확인:
                    # 필터 없이 돌리면 -71,270원(손익비 0.96)이다.


def get_htf(ticker):
    """1시간봉 지표 묶음. 데이터 부족/실패 시 None."""

    df = pyupbit.get_ohlcv(
        ticker,
        interval="minute60",
        count=120
    )

    if df is None or len(df) < MA_TREND + 5:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    current_price = float(close.iloc[-1])

    ma50 = float(
        close.rolling(MA_TREND).mean().iloc[-1]
    )

    # 직전 20봉(형성중인 현재봉 제외) 고점
    dc_high = float(
        high.iloc[-(DC_ENTRY + 1):-1].max()
    )

    # ATR(14)
    pc = close.shift()
    tr = (high - low).combine(
        (high - pc).abs(), max
    ).combine(
        (low - pc).abs(), max
    )
    atr = float(tr.rolling(ATR_N).mean().iloc[-1])

    # 24시간 모멘텀
    if len(close) > MOM_LOOKBACK:
        mom24 = float(
            current_price / close.iloc[-(MOM_LOOKBACK + 1)] - 1
        )
    else:
        mom24 = 0.0

    return {
        "ticker": ticker,
        "current_price": current_price,
        "ma50": ma50,
        "dc_high": dc_high,
        "atr": atr,
        "mom24": mom24,
    }


def get_btc_regime():
    """
    BTC 일봉 종가 > 일봉 ma50 이면 위험선호(True) → 알트 신규매수 허용.
    조회 실패 시 보수적으로 False(매수 중단).
    """

    df = pyupbit.get_ohlcv(
        "KRW-BTC",
        interval="day",
        count=80
    )

    if df is None or len(df) < REGIME_MA + 2:
        return False

    ma = df["close"].rolling(REGIME_MA).mean()

    # 마지막 '완성된' 일봉 기준 (iloc[-1]은 형성중)
    return bool(
        df["close"].iloc[-2] > ma.iloc[-2]
    )


if __name__ == "__main__":

    print("BTC 위험선호 국면:", get_btc_regime())

    d = get_htf("KRW-BTC")

    if d:
        print(d)
        print(
            "돌파?",
            d["current_price"] > d["dc_high"]
            and d["current_price"] > d["ma50"]
        )
