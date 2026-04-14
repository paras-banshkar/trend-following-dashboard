import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = {
    "Equity": {"Nifty 50": "^NSEI", "S&P 500": "SPY", "Nikkei": "^N225"},
    "Currency": {"USD/INR": "USDINR=X", "EUR/USD": "EURUSD=X"},
    "Commodity": {"Gold": "GC=F", "Crude Oil": "CL=F", "Silver": "SI=F"},
    "Crypto": {"BTC": "BTC-USD", "ETH": "ETH-USD"}
}

def fetch_data(ticker, period="2y"):
    df = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # flatten columns
    df.dropna(inplace=True)
    return df

def calculate_indicators(df):
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["High50"] = df["High"].rolling(50).max()
    df["Low50"] = df["Low"].rolling(50).min()
    # ATR (20-day)
    df["TR"] = np.maximum(df["High"] - df["Low"],
               np.maximum(abs(df["High"] - df["Close"].shift()),
                          abs(df["Low"] - df["Close"].shift())))
    df["ATR"] = df["TR"].rolling(20).mean()
    return df

def run_backtest(df, portfolio_value=10_000_000):  # 1 Cr = 10M INR
    df = df.copy().dropna()
    position = 0
    entry_price = 0
    trailing_stop = 0
    peak_since_entry = 0
    equity = portfolio_value
    equity_curve = []
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # TREND FILTER
        trend_up = bool(prev["SMA50"].item() > prev["SMA100"].item())
        trend_down = bool(prev["SMA50"].item() < prev["SMA100"].item())
        if position == 0:
            # LONG ENTRY: 50-day high breakout + trend filter
            if trend_up and prev["Close"] >= prev["High50"]:
                atr = prev["ATR"]
                units = (equity * 0.002) / atr if atr > 0 else 0
                position = units
                entry_price = row["Open"]
                peak_since_entry = entry_price
                trailing_stop = entry_price - 3 * atr

            # SHORT ENTRY
            elif trend_down and prev["Close"] <= prev["Low50"]:
                atr = prev["ATR"]
                units = (equity * 0.002) / atr if atr > 0 else 0
                position = -units
                entry_price = row["Open"]
                peak_since_entry = entry_price
                trailing_stop = entry_price + 3 * atr

        elif position > 0:  # IN LONG
            peak_since_entry = max(peak_since_entry, row["High"])
            trailing_stop = peak_since_entry - 3 * row["ATR"]
            if row["Low"] <= trailing_stop:
                pnl = (trailing_stop - entry_price) * position
                equity += pnl
                trades.append(pnl)
                position = 0

        elif position < 0:  # IN SHORT
            peak_since_entry = min(peak_since_entry, row["Low"])
            trailing_stop = peak_since_entry + 3 * row["ATR"]
            if row["High"] >= trailing_stop:
                pnl = (entry_price - trailing_stop) * abs(position)
                equity += pnl
                trades.append(pnl)
                position = 0

        equity_curve.append({"Date": row.name, "Equity": equity})

    return pd.DataFrame(equity_curve).set_index("Date"), trades

def calc_metrics(equity_df, trades):
    eq = equity_df["Equity"]
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    years = len(eq) / 252
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    wins = [t for t in trades if t > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2),
        "Total Trades": len(trades)
    }

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Trend Following Dashboard", layout="wide")
st.title("📈 Trend Following Backtest Dashboard")

# Flatten ticker list for sidebar
all_tickers = {f"{k} — {name}": sym
               for k, v in TICKERS.items()
               for name, sym in v.items()}

selected_label = st.sidebar.selectbox("Select Ticker", list(all_tickers.keys()))
ticker_sym = all_tickers[selected_label]

with st.spinner("Fetching data & running backtest..."):
    df = fetch_data(ticker_sym)
    df = calculate_indicators(df)
    equity_df, trades = run_backtest(df)
    metrics = calc_metrics(equity_df, trades)

# --- Current Signal ---
last = df.iloc[-1]
prev = df.iloc[-2]
signal = "⚪ FLAT"
if prev["SMA50"] > prev["SMA100"] and last["Close"] >= last["High50"]:
    signal = "🟢 LONG (Breakout)"
elif prev["SMA50"] < prev["SMA100"] and last["Close"] <= last["Low50"]:
    signal = "🔴 SHORT (Breakdown)"

st.sidebar.metric("Current Signal", signal)
st.sidebar.metric("ATR (20d)", round(float(last["ATR"]), 4))

# --- Metrics Row ---
cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, v)

# --- Equity Curve Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["Equity"],
                          name="Strategy Equity", line=dict(color="cyan")))
fig.update_layout(title=f"Equity Curve — {selected_label}",
                  xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
                  template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- Price + Signals Chart ---
fig2 = go.Figure()
fig2.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                               low=df["Low"], close=df["Close"], name="Price"))
fig2.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", line=dict(color="orange")))
fig2.add_trace(go.Scatter(x=df.index, y=df["SMA100"], name="SMA 100", line=dict(color="blue")))
fig2.update_layout(title="Price Chart with Trend Filter",
                   template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)
