"""퀀트 포트폴리오 백테스트 도구

노트북(Quant_Portfolio.ipynb)에서 불러 쓰는 함수 모음.
- 이동평균 교차 전략 (거래 비용 포함)
- 성과 지표 (CAGR, 변동성, 샤프, MDD)
- 포트폴리오 비중 계산 (최대 샤프 / 최소 분산 / 리스크 패리티 / 동일 비중)
- 워크포워드 백테스트: 각 리밸런싱 시점에 '그 시점까지의 데이터'만으로 결정
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE = 0.025          # 무위험 수익률 (연, 한국 기준금리 수준)
FEE = 0.00015              # 증권사 수수료 (매수·매도 각각)
SELL_TAX = 0.0020          # 증권거래세 (주식 매도 시, 2026년 코스피·코스닥 0.20%)


# ---------------------------------------------------------------------------
# 1. 이동평균 교차 전략
# ---------------------------------------------------------------------------
def ma_cross_returns(close: pd.Series, short=20, long=60, is_etf=False):
    """단기 이동평균 > 장기 이동평균이면 보유, 아니면 현금.

    - 오늘 종가로 만든 신호는 '다음 날' 포지션에 반영한다 (미래 정보 사용 방지).
    - 포지션이 바뀌는 날 거래 비용을 뺀다 (매수: 수수료, 매도: 수수료 + 거래세(주식만)).
    반환: (전략 일간 수익률, 포지션)
    """
    ret = close.pct_change()
    signal = (close.rolling(short).mean() > close.rolling(long).mean()).astype(float)
    position = signal.shift(1).fillna(0)
    change = position.diff().fillna(position)
    buy_cost = (change > 0) * FEE
    sell_cost = (change < 0) * (FEE + (0 if is_etf else SELL_TAX))
    strat = position * ret.fillna(0) - buy_cost - sell_cost
    return strat, position


# ---------------------------------------------------------------------------
# 2. 성과 지표
# ---------------------------------------------------------------------------
def metrics(returns: pd.Series, label=""):
    r = returns.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - RISK_FREE) / vol if vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return pd.Series({"CAGR": cagr, "변동성": vol, "샤프": sharpe, "MDD": mdd}, name=label)


def sharpe_of(r: pd.Series):
    vol = r.std()
    return -np.inf if vol == 0 or np.isnan(vol) else (r.mean() * TRADING_DAYS - RISK_FREE) / (vol * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# 3. 비중 계산 방법
# ---------------------------------------------------------------------------
def _optimize(objective, n, max_weight):
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bounds = [(0, max_weight)] * n
    res = minimize(objective, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons)
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def weights_max_sharpe(R: pd.DataFrame, max_weight=0.20):
    """기대수익률(과거 평균)과 공분산으로 샤프 지수를 최대화. 평균 추정 오차에 민감하다."""
    mu, cov = R.mean().values * TRADING_DAYS, R.cov().values * TRADING_DAYS
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - RISK_FREE) / vol
    return _optimize(neg_sharpe, R.shape[1], max_weight)


def weights_min_variance(R: pd.DataFrame, max_weight=0.20):
    """기대수익률 없이 공분산만으로 변동성을 최소화."""
    cov = R.cov().values * TRADING_DAYS
    return _optimize(lambda w: w @ cov @ w, R.shape[1], max_weight)


def weights_risk_parity(R: pd.DataFrame, max_weight=0.20):
    """각 종목이 포트폴리오 위험에 기여하는 정도를 같게 맞춘다 (기대수익률 사용 안 함)."""
    cov = R.cov().values * TRADING_DAYS
    n = R.shape[1]
    def objective(w):
        port_var = w @ cov @ w
        contrib = w * (cov @ w) / port_var
        return ((contrib - 1 / n) ** 2).sum()
    return _optimize(objective, n, max(max_weight, 1 / n))


def weights_equal(R: pd.DataFrame, max_weight=None):
    n = R.shape[1]
    return np.ones(n) / n


METHODS = {
    "최대샤프": weights_max_sharpe,
    "최소분산": weights_min_variance,
    "리스크패리티": weights_risk_parity,
    "동일비중": weights_equal,
}


# ---------------------------------------------------------------------------
# 4. 워크포워드 백테스트
# ---------------------------------------------------------------------------
def walk_forward(hold_ret: pd.DataFrame, ma_ret: pd.DataFrame, etf_flags: dict,
                 method="최대샤프", lookback=252, rebalance=63, select_ma=True, max_weight=0.20):
    """리밸런싱 시점마다 직전 lookback일 데이터만 보고 결정하고, 다음 시점까지 그대로 운용한다.

    hold_ret : 종목별 단순 보유 일간 수익률
    ma_ret   : 종목별 이동평균 전략 일간 수익률 (거래 비용 포함)
    select_ma: True면 종목마다 과거 구간 샤프가 높은 쪽(전략/보유)을 고른다.
               (원래 노트북은 전체 기간으로 골라 미래 정보가 섞였던 부분)
    - 리밸런싱 사이에는 가격 변화에 따라 비중이 자연스럽게 변한다.
    - 리밸런싱할 때 바뀐 비중만큼 거래 비용을 뺀다.
    반환: (일간 수익률, 리밸런싱 기록 DataFrame)
    """
    tickers = hold_ret.columns
    sell_cost = np.array([FEE + (0 if etf_flags[t] else SELL_TAX) for t in tickers])
    dates = hold_ret.index
    starts = list(range(lookback, len(dates), rebalance))
    out, log = [], []
    w_prev = np.zeros(len(tickers))          # 첫 시점은 현금에서 시작
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(dates)
        past_h, past_m = hold_ret.iloc[s - lookback:s], ma_ret.iloc[s - lookback:s]
        if select_ma:
            use_ma = np.array([sharpe_of(past_m[t]) > sharpe_of(past_h[t]) for t in tickers])
        else:
            use_ma = np.zeros(len(tickers), bool)
        past = pd.DataFrame(np.where(use_ma, past_m.values, past_h.values), index=past_h.index, columns=tickers)
        w = METHODS[method](past, max_weight)
        # 리밸런싱 비용 (매수: 수수료, 매도: 수수료 + 거래세)
        diff = w - w_prev
        cost = (np.clip(diff, 0, None) * FEE).sum() + (np.clip(-diff, 0, None) * sell_cost).sum()
        fut = np.where(use_ma, ma_ret.iloc[s:e].values, hold_ret.iloc[s:e].values)
        # 다음 리밸런싱까지 비중이 가격 변화에 따라 변한다
        cur, rets = w.copy(), []
        for i, r in enumerate(fut):
            pr = cur @ r - (cost if i == 0 else 0)
            rets.append(pr)
            cur = cur * (1 + r)
            cur = cur / cur.sum() if cur.sum() > 0 else cur
        out.append(pd.Series(rets, index=dates[s:e]))
        w_prev = cur
        log.append({"date": dates[s], "비용": cost, "전략사용": int(use_ma.sum()),
                    **{t: wi for t, wi in zip(tickers, w)}})
    return pd.concat(out), pd.DataFrame(log).set_index("date")
