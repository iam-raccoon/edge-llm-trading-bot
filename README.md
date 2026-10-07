# Xavier NX AI — 로컬 LLM 기반 주식 뉴스 봇 & 코인 자동매매 봇

NVIDIA **Jetson Xavier NX** 한 대 위에서 **로컬 LLM(llama.cpp + Qwen2.5)** 을 직접 서빙하고,
그 위에 **주식 뉴스 요약 봇**과 **코인 자동매매 봇**을 올린 온디바이스 개인 AI 비서입니다.

> 외부 LLM API(OpenAI 등) 없이, 모델 추론까지 전부 이 작은 보드 안에서 돌아갑니다.
> 뉴스 수집·지표 계산·매매 판단·텔레그램 알림이 하나의 파이프라인으로 연결됩니다.

🏭 **실전 배포(공장)** — 여기 올라간 매매 전략의 **백테스트·엣지 검증(연구실)은 → [llm-trading-backtest](https://github.com/iam-raccoon/llm-trading-backtest)**. 저기서 "진짜 엣지 있나" 검증 → 여기서 라이브 실행.

---

## 목차
1. [한눈에 보는 구조](#1-한눈에-보는-구조)
2. [작동 원리 ① 로컬 LLM 서빙](#2-작동-원리--로컬-llm-서빙)
3. [작동 원리 ② 주식 뉴스 봇](#3-작동-원리--주식-뉴스-봇)
4. [작동 원리 ③ 코인 자동매매 봇](#4-작동-원리--코인-자동매매-봇)
5. [작동 원리 ④ 텔레그램 봇 & 스케줄](#5-작동-원리--텔레그램-봇--스케줄)
6. [폴더 구조](#6-폴더-구조)
7. [설치 & 실행](#7-설치--실행)
8. [기술 스택](#8-기술-스택)
9. [주의사항 (면책)](#9-주의사항-면책)

---

## 1. 한눈에 보는 구조

```
                        ┌───────────────────────── Jetson Xavier NX ──────────────────────────┐
                        │                                                                      │
   Naver 뉴스 API ──────┼──▶  news / market / stock ──┐                                        │
                        │                              │  요약·판단 요청                        │
   Upbit 시세/주문 ─────┼──▶  crypto (auto_trader) ────┼──▶  llm_client ──▶ llama-server :8080  │
                        │                              │                    (Qwen2.5-1.5B)     │
   yfinance 주가 ───────┼──▶  stock / market ──────────┘                                        │
                        │                                                                      │
                        │  결과 알림/리포트  ──▶  bot (telegram)                                 │
                        └──────────────────────────────────────────────┬───────────────────────┘
                                                                        │
                                                                        ▼
                                                                   Telegram (사용자)
```

핵심 아이디어는 **"모든 모듈이 동일한 로컬 LLM 엔드포인트를 공유한다"** 입니다.
`llama-server` 가 OpenAI 호환 API(`/v1/chat/completions`)를 `localhost:8080` 에 열어두면,
주식·코인·뉴스 어느 모듈이든 `llm/llm_client.py` 하나로 똑같이 호출합니다.

---

## 2. 작동 원리 ① 로컬 LLM 서빙

**목표:** 클라우드 API 없이, 엣지 디바이스에서 LLM 추론을 상시 제공한다.

1. **모델** — `Qwen2.5-1.5B-Instruct` 를 `Q4_K_M`(4비트 양자화, GGUF) 로 사용.
   8GB 메모리의 Xavier NX 에서도 GPU 로 돌릴 수 있을 만큼 가볍습니다.
2. **서빙** — `llama.cpp` 를 `GGML_CUDA=ON` 으로 빌드해 Jetson GPU(sm_72)에서 구동.
   `llama-server` 가 **OpenAI 호환** 챗 API 를 `127.0.0.1:8080` 에 노출합니다.
3. **클라이언트** — [`llm/llm_client.py`](llm/llm_client.py) 의 `ask_llm(prompt)` 가
   시스템 프롬프트([`llm/system_prompt.txt`](llm/system_prompt.txt), 페르소나 'Xavier')와
   사용자 프롬프트를 묶어 `POST /v1/chat/completions` 로 보냅니다.
   (`temperature=0.2`, `max_tokens=192` — 짧고 일관된 답을 위해)

```python
# 모든 모듈이 이 한 줄로 로컬 LLM을 씁니다
from llm.llm_client import ask_llm
opinion = ask_llm("삼성전자 추세는 상승, RSI 중립, 뉴스 긍정. 두 문장으로 의견을 줘.")
```

> 즉, LLM 은 "판단을 대신 내리는 주체"가 아니라 **정량 지표를 사람이 읽기 좋은 문장으로 바꾸는 요약기**로 씁니다.
> 매수/매도 같은 실제 결정은 아래의 규칙 기반 로직이 담당합니다.

---

## 3. 작동 원리 ② 주식 뉴스 봇

주가 데이터(yfinance)와 뉴스(Naver)를 합쳐, **추천 종목**과 **개별 종목 분석**을 만들어냅니다.

### (a) 뉴스 수집 & 감성 분석 — [`news/news_api.py`](news/news_api.py)
- Naver 뉴스 검색 API 로 키워드별 최신 기사 제목을 가져옵니다.
- 제목에 **긍정 단어**(증가·성장·호조·흑자·수주…)와 **부정 단어**(감소·하락·적자·급락·부진…)가
  몇 번 나오는지 세어 `긍정 / 중립 / 부정` 으로 분류합니다. (가볍고 빠른 키워드 기반 감성분석)

### (b) 종목 스코어링 — [`market/recommend_stock.py`](market/recommend_stock.py)
각 종목의 3개월치 주가로 지표를 계산해 점수를 매깁니다.

| 지표 | 점수 규칙 |
|---|---|
| 추세 | `현재가 > MA20 > MA60` → +40 (상승) / `현재가 > MA60` → +20 (횡보) |
| RSI(14) | 40~65 → +20 / 30~70 → +10 |
| 거래량 | 20일 평균 대비 증가 → +20 |
| 뉴스 | 긍정 → +20 / 중립 → +10 |

- 상위 종목에 대해 추세에 따라 **목표가(+5~12%)** 와 **손절가(−5~10%)** 를 자동 산출.
- 최종 점수 상위 3종목을 데일리 리포트에 싣습니다.

### (c) 개별 종목 대화 분석 — [`stock/stock_analyzer.py`](stock/stock_analyzer.py)
텔레그램에 "삼성전자" 처럼 종목명을 보내면:
`추세 판단 → RSI 상태 → 뉴스 분위기 → 추천 매수가/목표가/손절가` 를 계산하고,
마지막에 **LLM 이 두 문장짜리 투자 의견**을 덧붙여 돌려줍니다.

### (d) 데일리 시장 브리핑 — [`daily_market_report.py`](daily_market_report.py)
`미국 뉴스 + 국내 뉴스 + USD/KRW 환율 + 추천 종목 3개 + LLM 한 줄 시황 코멘트`
를 하나의 리포트로 조립해 아침에 텔레그램으로 발송합니다.

---

## 4. 작동 원리 ③ 코인 자동매매 봇

Upbit 원화마켓 코인을 대상으로, **다중 지표 스코어링 → 규칙 기반 매수/매도** 를 반복합니다.
대상 코인 목록은 [`crypto/coins.json`](crypto/coins.json) 에서 관리합니다.

> 🆕 **v1.1.1 안내** — 아래 **(a)~(e)** 는 5분봉 스코어링 방식(**v1.0 레거시**)입니다.
> 실거래 로그와 백테스트로 이 방식이 **우위(edge)가 없음**을 확인한 뒤,
> **현재 활성 매매 엔진은 [(f) 1시간봉 추세추종](#f--v111--1시간봉-추세추종-엔진-백테스트-검증)** 으로 전면 교체했습니다.
> (`crypto_bot.service` 가 구동하는 것은 이제 `auto_trader_htf.py` 입니다.)

### (a) 지표 계산 — [`crypto/indicators.py`](crypto/indicators.py)
5분봉(단기)과 4시간봉(추세)을 함께 봅니다:
`MA20 / MA60`, `RSI(14)`, `거래량 비율`, `최근 1시간 수익률`, `볼린저밴드`, `ATR(14)`, `MACD(12·26·9)`, `4시간봉 추세`.

### (b) 점수화 — [`crypto/coin_analyzer.py`](crypto/coin_analyzer.py)
위 지표를 합산해 **−∞ ~ +100점대**의 점수와 추세(상승/횡보/하락)를 냅니다.
예: 정배열이면 +40, RSI 45~60이면 +30, 거래량 급증이면 +20, MACD 골든이면 +10, 과열(RSI>75)이면 큰 감점.

### (c) 랭킹 — [`crypto/coin_ranker.py`](crypto/coin_ranker.py)
후보 코인 전체를 점수순으로 정렬해 **상위 10개**를 뽑습니다.

### (d) 매매 엔진 — [`crypto/trade_manager.py`](crypto/trade_manager.py)
`manage_trade()` 가 한 사이클마다 다음을 수행합니다.

**보유 종목 관리 (매도 규칙)**

| 조건 | 동작 |
|---|---|
| 수익률 **≥ +3%** | 익절 매도 🔴 |
| 수익률 **≤ −4%** | 손절 매도 🔵 |
| 보유 **72시간** 초과 | 수익 중이면 시간만료 매도 ⏰, 아니면 보유 연장 |
| "하락" 추세 **5회 연속** | 추세붕괴 매도 ⚠️ |

**신규 매수 규칙**
- 동시 보유 **최대 3종목**, 종목당 **5만원**
- 점수 **≥ 80**, 거래량비율 **≥ 0.8**, RSI **< 70** 을 모두 만족할 때만 진입

### (e) 상태 저장 & 성과 분석
- 보유 포지션 → `crypto/position.json`, 매매 기록 → `crypto/trade_log.json` 로 영속화
  (둘 다 개인 거래기록이라 `.gitignore` 처리 — 저장소에 올라가지 않음)
- [`crypto/performance.py`](crypto/performance.py) 가 **승률 · 평균 수익률 · 누적 수익률 · 최대 손익**을 계산
- [`crypto/portfolio_manager.py`](crypto/portfolio_manager.py) 가 현재 평가금·수익률을 텔레그램용으로 정리

### ⚠️ 현재는 "가상매매(페이퍼 트레이딩)" 모드
[`crypto/upbit_api.py`](crypto/upbit_api.py) 의 실제 주문 코드(`buy_market_order` / `sell_market_order`)는
**주석 처리**되어 있고, 지금은 매수/매도를 시뮬레이션만 합니다.
전략을 실계좌 없이 안전하게 검증하기 위한 기본값이며, 실거래 전환은 해당 주석을 해제하면 됩니다.

---

### (f) 🆕 v1.1.1 — 1시간봉 추세추종 엔진 (백테스트 검증)

**왜 바꿨나.** (a)~(e)의 5분봉 스코어링 봇을 실거래 로그(57건)로 뜯어보니, 전체 청산의
**89%가 `추세붕괴`** 였고 승률 14% / 누적 **−30%** 였습니다. 4개월치 과거 데이터로 백테스트한 결과
**5분봉 기반으로는 진입 신호 자체에 우위가 없었습니다** — BTC가 −1.4%(횡보)인 구간에서도 전략은
−17~24%. 5분봉은 추세보다 노이즈가 커서, 수수료(왕복 0.1%)를 이기지 못합니다.
→ 5분봉을 버리고 **상위 타임프레임(1시간봉) 추세추종**으로 재설계했습니다.

**알고리즘 = 시장국면 필터 + Donchian 돌파 진입 + 샹들리에 트레일링 청산**

**① 시장 국면 필터 (진입 게이트)** — [`crypto/htf_indicators.py`](crypto/htf_indicators.py) `get_btc_regime()`
> **BTC 일봉 종가 > BTC 일봉 50일선** 일 때만 "위험선호(risk-on)"로 보고 신규 매수를 허용.
> BTC가 50일선 아래면 알트는 전부 **현금 보유**. 알트코인은 BTC의 고베타(high-beta) 자산이라,
> **시장 전체가 위험선호일 때만** 베팅하는 것이 핵심입니다. (하락장 방어의 90%가 이 한 줄에서 나옴)

**② 진입 — 20봉 고점 돌파 (Donchian Breakout)** — [`crypto/htf_manager.py`](crypto/htf_manager.py)
> 각 코인의 현재가가 **직전 20개 1시간봉의 최고가를 상향 돌파**하고, 동시에 **50시간 이동평균 위**에
> 있으면 "추세 시작"으로 보고 매수 후보에 올립니다. 후보가 여럿이면 **24시간 모멘텀이 강한 순**으로
> 최대 3종목(종목당 5만원)을 잡습니다. → *강세장에서 새 고점을 뚫는 놈에 올라탄다.*

**③ 청산 — 샹들리에 트레일링 스탑 (Chandelier Exit, 3×ATR)**
> 진입 후 **최고가에서 3×ATR(14) 만큼 아래**에 스탑을 겁니다. 가격이 오르면 스탑도 따라 올라가고,
> 고점 대비 3×ATR 떨어지면 청산. → **손실은 변동성 기반의 진짜 손절선으로 짧게 자르고,
> 수익은 추세가 살아있는 한 상한 없이 태웁니다.** (5분봉의 노이즈 청산 `추세붕괴`를 폐기한 이유)

**백테스트 성과 (2026-03-19 ~ 07-14 · 118일 · 16개 코인 · 수수료 0.1% 반영)**

| 구분 | 성과 |
|---|---|
| BTC 단순보유 (벤치마크) | **−12.2%** (기간 내내 하락장) |
| 16개 알트 평균 단순보유 | **−22.9%** |
| **본 전략 (HTF 추세추종)** | **+25.3%** (구간 의존 — 아래 ⚠️ 참고) |

- 승률은 38%지만 **평균수익 +3.8% vs 평균손실 −1.9% (손익비 ≈ 2.0)**, 최대손실 −5.4%로 리스크가 타이트.
- 돌파 봉수(15~25)·ATR 배수(2.5~3.5) 전 조합에서 플러스 → 특정 값 **과최적화 아님**.
- **하락장에서 시장을 37~48%p 이겼습니다.** 수익은 BTC가 위험선호로 돌아선 구간에 집중되고,
  침체기엔 국면필터가 현금을 지켜 손실을 방어합니다.

> ### ⚠️ 이 +25.3%를 그대로 믿지 말 것 — 구간에 따라 부호가 뒤집힙니다
>
> 같은 HTF 전략을 연구 저장소([llm-trading-backtest](https://github.com/iam-raccoon/llm-trading-backtest))에서
> **다른 창·다른 코인 구성**으로 다시 재보면 숫자가 이렇게 달라집니다.
>
> | 측정 | 창 | 결과 |
> |---|---|---|
> | 이 문서 (118일 · 16종) | 2026-03-19 ~ 07-14 | **+25.3%** |
> | RESULTS.md 5장 (90일) | — | **+2.19%** |
> | RESULTS.md 7장 | 2026-05-07 ~ 08-03 | **−1.05%** |
>
> 세 수치는 기간·코인수·비교대상이 달라 **같은 척도로 비교할 수 없습니다.** 다만 방향은 분명합니다 —
> **돌파(breakout) 전략은 창이 며칠만 밀려도 부호가 뒤집힐 만큼 구간에 민감합니다.**
> 위 +25.3%는 "이 전략의 실력"이 아니라 **"저 118일 창에서의 결과"** 로 읽어야 합니다.

**구성 파일**
| 파일 | 역할 |
|---|---|
| [`crypto/htf_indicators.py`](crypto/htf_indicators.py) | 1h 지표(ma50·돌파선·ATR·모멘텀) + BTC 일봉 국면 판정 |
| [`crypto/htf_manager.py`](crypto/htf_manager.py) | 돌파 진입 · 샹들리에 청산 · 국면필터 · 별도 로깅 |
| [`crypto/auto_trader_htf.py`](crypto/auto_trader_htf.py) | 1시간 주기 실행 루프 (`crypto_bot.service` 가 구동) |
| [`crypto/htf_performance.py`](crypto/htf_performance.py) | 페이퍼 성과 모니터 |

> ⚠️ **아직 페이퍼(가상매매)입니다.** 검증은 과거 1사이클(4개월)뿐이라 과최적화 위험이 남아 있어,
> 실거래 전환 전 반드시 **실시간 아웃오브샘플(포워드) 검증**을 거칩니다.
> 포지션/로그는 5분봉 봇과 분리된 `position_htf.json` / `trade_log_htf.json` 에 기록됩니다(gitignore).

---

## 5. 작동 원리 ④ 텔레그램 봇 & 스케줄

### 텔레그램 봇 — [`telegram_bot.py`](telegram_bot.py)
`python-telegram-bot` 으로 폴링하며, 메시지 종류에 따라 분기합니다.

| 입력 | 동작 |
|---|---|
| `/status` | 봇·LLM 서버 상태 표시 |
| `/brief` 또는 `오늘 브리핑` | 데일리 시장 브리핑 생성 |
| `/추천` | AI 추천 종목 Top3 |
| `/coin` | 코인 포트폴리오 현황 |
| 종목명 (예: `삼성전자`) | 개별 종목 분석 |
| 그 외 자유 문장 | 로컬 LLM 일반 대화 |

발송 전용 유틸은 [`bot/telegram_sender.py`](bot/telegram_sender.py) 의 `notify()` 로,
매매 알림([`crypto/alarm_manager.py`](crypto/alarm_manager.py))이 이걸 통해 텔레그램에 알립니다.

### 상시 가동 (systemd + cron)
- **systemd** — `llama-server` 를 부팅 시 상시 구동
- **cron** — 평일 아침 `send_daily_report.py` 실행해 브리핑 자동 발송
- **crypto_bot.service** — [`crypto/auto_trader_htf.py`](crypto/auto_trader_htf.py) 가 1시간 주기로 `manage_htf()` 반복 (v1.1.1 추세추종 엔진, 페이퍼)

---

## 6. 폴더 구조

```
xavier_nx_ai/
├── llm/                    # 로컬 LLM 클라이언트 (llama-server 호출)
│   ├── llm_client.py
│   └── system_prompt.txt   # 페르소나 'Xavier' 시스템 프롬프트
├── news/                   # Naver 뉴스 수집 + 키워드 감성분석
├── market/                 # 미국/국내 뉴스 · 환율 · 종목 추천
├── stock/                  # 개별 종목 분석 (yfinance)
├── crypto/                 # 코인 자동매매 (Upbit)
│   │  # ── 🆕 v1.1.1 활성 엔진: 1시간봉 추세추종 ──
│   ├── htf_indicators.py   #   1h 지표 + BTC 일봉 국면
│   ├── htf_manager.py      #   돌파 진입 · 샹들리에 청산
│   ├── auto_trader_htf.py  #   1시간 주기 루프 (crypto_bot.service)
│   ├── htf_performance.py  #   페이퍼 성과 모니터
│   │  # ── v1.0 레거시: 5분봉 스코어링 ──
│   ├── indicators.py       #   지표 계산
│   ├── coin_analyzer.py    #   점수화
│   ├── coin_ranker.py      #   랭킹
│   ├── trade_manager.py    #   매수/매도 엔진
│   ├── auto_trader.py      #   1시간 주기 루프
│   ├── portfolio_manager.py#   포트폴리오 조회
│   └── performance.py      #   성과 분석
├── bot/                    # 텔레그램 발송 유틸
├── daily_market_report.py  # 데일리 브리핑 조립
├── send_daily_report.py    # 브리핑 발송 (cron)
├── telegram_bot.py         # 텔레그램 대화 봇 (진입점)
├── secrets_example/        # API 키 템플릿 (실제 키는 secrets/, gitignore)
└── docs/                   # 셋업 문서 시리즈
```

---

## 7. 설치 & 실행

### 1) 의존성
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) API 키 설정
실제 키는 저장소에 포함되지 않습니다. 템플릿을 복사해 `secrets/` 에 채워 넣으세요.
```bash
mkdir -p secrets
cp secrets_example/upbit_key.example.json   secrets/upbit_key.json
cp secrets_example/naver_api.example.json   secrets/naver_api.json
cp secrets_example/telegram.example.json    secrets/telegram.json
# 각 파일에 실제 키 입력 (secrets/ 는 .gitignore 처리됨)
```

### 3) 로컬 LLM 서버 실행
`llama.cpp` 를 CUDA 로 빌드한 뒤 `llama-server` 를 `:8080` 에 띄웁니다.
(자세한 과정은 [`docs/01-llm-setup.md`](docs/01-llm-setup.md) 참고)

### 4) 실행
```bash
python telegram_bot.py          # 텔레그램 대화 봇
python send_daily_report.py     # 데일리 브리핑 1회 발송
python crypto/auto_trader_htf.py   # 코인 자동매매 (v1.1.1 추세추종, 가상매매)
python crypto/htf_performance.py   # 매매 성과 확인
```

---

## 8. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 디바이스 | NVIDIA Jetson Xavier NX (8GB, sm_72) |
| OS / 런타임 | L4T R35.6 (JetPack 5.x), CUDA 11.4, Python 3.8 |
| LLM 서빙 | llama.cpp (GGML_CUDA=ON), llama-server |
| 모델 | Qwen2.5-1.5B-Instruct (Q4_K_M, GGUF) |
| 데이터 | Upbit API, Naver 뉴스 API, yfinance |
| 봇/알림 | python-telegram-bot |
| 운영 | systemd, cron |

---

## 9. 주의사항 (면책)

- 이 프로젝트는 **개인 학습·실험용**입니다. 투자 자문이나 수익을 보장하지 않습니다.
- 코인 매매 로직은 기본적으로 **가상매매** 이며, 실거래 전환 시 발생하는 모든 손실은 사용자 책임입니다.
- 뉴스 감성분석·종목 점수는 단순 규칙 기반이므로, 실제 투자 판단의 근거로 삼지 마세요.
- API 키 등 민감정보는 절대 커밋하지 마세요. (`secrets/` 는 `.gitignore` 로 보호됩니다)
