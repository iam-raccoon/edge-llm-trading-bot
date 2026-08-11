"""
1시간봉 추세추종 매매 관리 (페이퍼/실거래 공용).

기존 5분봉 봇(trade_manager.py)과 완전히 분리:
  - 포지션 파일 : position_htf.json
  - 거래 로그   : trade_log_htf.json

실거래는 upbit_api.buy_coin/sell_coin 이 가상매매 스텁이라 현재 페이퍼 모드.
"""

import json
import sys
import time
import datetime as dt

sys.path.append("/home/user/xavier_nx_ai")

from crypto.htf_indicators import get_htf, get_btc_regime
from crypto.upbit_api import buy_coin, sell_coin
from crypto.alarm_manager import buy_alarm, sell_alarm

with open(
    "/home/user/xavier_nx_ai/crypto/coins.json",
    encoding="utf-8"
) as f:
    COINS = json.load(f)

POSITION_FILE = "/home/user/xavier_nx_ai/crypto/position_htf.json"
LOG_FILE = "/home/user/xavier_nx_ai/crypto/trade_log_htf.json"

BUY_AMOUNT = 50000
MAX_POSITION = 3
CHAND_MULT = 3.0        # 샹들리에 트레일링: 최고가 - 3×ATR
FEE = 0.001             # 왕복 수수료 0.1% (실손익 반영)


# ============ 입출력 ============
def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            if data.strip() == "":
                return default
            return json.loads(data)
    except Exception:
        return default


def load_positions():
    return _load_json(POSITION_FILE, {})


def save_positions(pos):
    with open(POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(pos, f, ensure_ascii=False, indent=4)


def save_trade(ticker, buy_price, sell_price, hold_hours, reason,
               buy_time=None):
    """청산 기록. **시각을 반드시 남긴다** — 없으면 기간별·국면별 분석이
    영원히 불가능하다(2026-08-10 이전 13건이 그래서 날짜를 잃었다)."""
    logs = _load_json(LOG_FILE, [])
    profit = (sell_price / buy_price - 1) * 100 - FEE * 100
    now = dt.datetime.now()
    logs.append({
        "ticker": ticker,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "profit": round(profit, 2),
        "hold_hours": round(hold_hours, 1),
        "reason": reason,
        "sell_time": now.isoformat(timespec="seconds"),
        "buy_time": (dt.datetime.fromtimestamp(buy_time).isoformat(timespec="seconds")
                     if buy_time else None),
    })
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)


# ============ 매매 관리 ============
def manage_htf():

    positions = load_positions()
    remove_list = []

    # ---------- 보유 종목: 샹들리에 트레일링 청산 ----------
    for ticker in positions:

        pos = positions[ticker]
        data = get_htf(ticker)

        if data is None:
            continue

        current_price = data["current_price"]
        atr = data["atr"]

        buy_price = pos["buy_price"]
        quantity = pos.get("quantity")

        # 진입 후 최고가 갱신 (트레일링 기준)
        peak_high = max(
            pos.get("peak_high", buy_price),
            current_price
        )
        pos["peak_high"] = peak_high

        stop_price = peak_high - CHAND_MULT * atr

        profit = current_price / buy_price - 1
        hold_hours = (time.time() - pos["buy_time"]) / 3600

        print(
            ticker,
            round(profit * 100, 2), "%",
            "| peak",
            round((peak_high / buy_price - 1) * 100, 2), "%",
            "| stop",
            round((stop_price / buy_price - 1) * 100, 2), "%"
        )

        # 종가가 트레일링 스탑 이탈 → 청산
        if current_price <= stop_price:

            sell_coin(ticker, quantity)

            save_trade(
                ticker, buy_price, current_price,
                hold_hours, "📉 트레일청산",
                buy_time=pos.get("buy_time")
            )

            sell_alarm(
                ticker, buy_price, current_price,
                profit - FEE, hold_hours, "📉 트레일청산"
            )

            remove_list.append(ticker)

    for ticker in remove_list:
        del positions[ticker]

    save_positions(positions)

    # ---------- 신규 매수 (BTC 위험선호 국면에서만) ----------
    holding = list(positions.keys())

    if len(holding) >= MAX_POSITION:
        save_positions(positions)
        return

    if not get_btc_regime():
        print("BTC 위험회피 국면 → 신규 매수 중단 (현금 보유)")
        save_positions(positions)
        return

    # 돌파 후보 수집
    candidates = []

    for ticker in COINS:

        if ticker in positions:
            continue

        data = get_htf(ticker)

        if data is None:
            continue

        breakout = (
            data["current_price"] > data["dc_high"]
            and data["current_price"] > data["ma50"]
        )

        if breakout:
            candidates.append(data)

    # 24시간 모멘텀 강한 순
    candidates.sort(key=lambda x: x["mom24"], reverse=True)

    for data in candidates:

        if len(holding) >= MAX_POSITION:
            break

        ticker = data["ticker"]
        current_price = data["current_price"]

        quantity = round(BUY_AMOUNT / current_price, 6)

        buy_coin(ticker, BUY_AMOUNT)

        # score 자리에 24h 모멘텀(%) 표시
        buy_alarm(
            ticker,
            round(data["mom24"] * 100, 1),
            current_price,
            BUY_AMOUNT
        )

        positions[ticker] = {
            "buy_price": current_price,
            "buy_time": time.time(),
            "quantity": quantity,
            "peak_high": current_price,
        }

        holding.append(ticker)

    save_positions(positions)


if __name__ == "__main__":
    manage_htf()
