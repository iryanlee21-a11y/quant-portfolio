"""가격 데이터 수집 (pykrx). 결과를 data/prices.csv로 저장해 두고 노트북은 이 파일만 읽는다.
기간을 고정해 두었으므로 누가 언제 실행해도 같은 데이터로 같은 결과를 얻는다.
실행: python data/fetch_prices.py   (Colab에서는 !pip install pykrx 후 실행)"""
import pandas as pd
from pykrx import stock

START, END = "20240901", "20261030"     # 실제 투자 시작 시점 ~ 분석 시점
TICKERS = ["360200", "458730", "323280", "108490", "329180", "012450", "261240",
           "005930", "047040", "064350", "133690", "319640", "000660",
           "069500"]                     # 069500: KODEX 200 (KOSPI200 비교용)

close = {t: stock.get_market_ohlcv(START, END, t)["종가"] for t in TICKERS}
prices = pd.DataFrame(close)
prices.index.name = "date"
prices.to_csv("data/prices.csv")
print(prices.shape, prices.index.min().date(), "~", prices.index.max().date())
