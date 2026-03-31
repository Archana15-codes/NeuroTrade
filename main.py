from indicators import add_all_indicators
import yfinance as yf

df = yf.download("RELIANCE.NS", period="6mo")

df = add_all_indicators(df)

print(df.tail())