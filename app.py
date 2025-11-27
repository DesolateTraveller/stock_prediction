# app.py — White Background + Navy Cards (Professional Trading Dashboard)
#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Login
#---------------------------------------------------------------------------------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def authenticate(username, password):
    # Replace with real auth (e.g., database, OAuth)
    return username == "admin" and password == "password123"

if not st.session_state.logged_in:
    st.markdown('<div class="title-header">🔐 Please Login</div>', unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()  # Stop rendering rest of app
    
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

import smtplib
from email.mime.text import MIMEText
#---------------------------------------------------------------------------------------------------------------------------------
### Import Functions
#---------------------------------------------------------------------------------------------------------------------------------
from data_fetcher import fetch_nse_data, fetch_stock_history
from ml_predictor import train_and_predict_next_close
from strategies import add_technical_indicators, get_buy_recommendations, get_sell_recommendations

#---------------------------------------------------------------------------------------------------------------------------------
### Title for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Stock Dashboard | v1.0",
                   layout="wide",
                   page_icon="📈",              
                   initial_sidebar_state="auto")

#---------------------------------------------------------------------------------------------------------------------------------
### CSS
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #ffffff;
        color: #1e293b;
    }
    
    /* Header title */
    .title-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric cards — Dark Navy */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #334155;
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        background-color: #f8fafc !important; /* Light gray header */
        border-radius: 8px !important;
        padding: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8fafc;
        text-align: center;
        padding: 8px;
        font-size: 13px;
        color: #475569;
        z-index: 100;
        border-top: 1px solid #cbd5e1;
    }
    
    .footer a {
        color: #0072ff;
        text-decoration: none;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    
    /* Dataframe */
    .dataframe {
        font-size: 0.9rem !important;
        background-color: #ffffff !important;
    }
    
    /* Plotly chart container */
    .plotly-graph-div {
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Header
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Stock Dashboard</div>
    <div class="title-small">v1.0</div>
    """,
    unsafe_allow_html=True
)
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Main Layout
#---------------------------------------------------------------------------------------------------------------------------------
#col_sidebar, col_main = st.columns([0.25, 0.75])

if st.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
        
# Initialize session state for watchlist (do this once at the top of app.py if not done)
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

col1, col2, col3 = st.columns((0.3,0.4,0.3))

with col1:
    df = fetch_nse_data()
    
    with st.expander("📈 Stock Selection", expanded=True):
        selected_from_dropdown = st.selectbox(
            "🔍 Select Stock",
            options=[""] + sorted(df["Symbol"].tolist()),
            index=0,
            key="stock_selector"
        )
    
with col2:
    
    with st.expander("📌 My Watchlist", expanded=True):
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            add_stock = st.selectbox(
                "➕ Add to Watchlist",
                options=[""] + sorted(df["Symbol"].tolist()),
                index=0,
                key="watchlist_add"
            )
            if st.button("✅ Add", key="add_to_watchlist", use_container_width=True):
                if add_stock and add_stock not in st.session_state.watchlist:
                    st.session_state.watchlist.append(add_stock)
                    st.rerun()
                
        with subcol2:
            st.markdown("**Saved Stocks:**")
            if st.session_state.watchlist:
                for i, stock in enumerate(st.session_state.watchlist):
                    wl_col1, wl_col2 = st.columns([4, 1])
                    with wl_col1:
                        # Clicking this button selects the stock
                        if st.button(stock, key=f"select_{i}", use_container_width=True):
                            st.session_state.selected_stock = stock
                    with wl_col2:
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state.watchlist.remove(stock)
                            # Clear selection if removed stock was active
                            if hasattr(st.session_state, 'selected_stock') and st.session_state.selected_stock == stock:
                                del st.session_state.selected_stock
                            st.rerun()
            else:
                st.info("No stocks saved yet.")

    if hasattr(st.session_state, 'selected_stock'):
        selected_stock = st.session_state.selected_stock
    else:
        selected_stock = selected_from_dropdown

with col3:
    
    if selected_stock:
        with st.expander("📅 Timeframe", expanded=True):
            timeframe = st.selectbox(
                "Chart Interval",
                options=["1d", "5m", "15m", "1h", "1wk"],
                index=0,
                key="timeframe_selector"
            )
            period_map = {"5m": "60d", "15m": "60d", "1h": "730d", "1d": "3mo", "1wk": "2y"}
            period = period_map[timeframe]
    else:
        st.info("👈 Select a stock from dropdown or your watchlist to begin analysis.")

st.divider()

#with col_main:
if not selected_stock:

        with st.expander("🏆 Top Performers Today", expanded=True):
            
            top_gainers = df.head(5)
            st.dataframe(
                top_gainers.style.format({
                    "Open": "₹{:.2f}",
                    "High": "₹{:.2f}",
                    "Low": "₹{:.2f}",
                    "Close": "₹{:.2f}",
                    "Change %": "{:.2f}%"
                }),
                use_container_width=True
            )
        
        with st.expander("🎯 Smart Recommendations", expanded=True):
            
            buy_candidates = get_buy_recommendations(df, top_n=10)
            ml_buy_list = []
            for symbol in df["Symbol"].head(10):
                try:
                    hist = fetch_stock_history(symbol, period="3mo")
                    pred, _ = train_and_predict_next_close(hist)
                    if pred and pred["Signal"] == "BUY":
                        ml_buy_list.append({
                            "Symbol": symbol,
                            "Predicted_Change_%": pred["Expected_Change_%"]
                        })
                except:
                    continue

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("🟢 **ML BUY Signals**")
                    if ml_buy_list:
                        ml_buy_df = pd.DataFrame(ml_buy_list).sort_values("Predicted_Change_%", ascending=False)
                        st.dataframe(ml_buy_df.style.format({"Predicted_Change_%": "{:.2f}%"}), use_container_width=True)
                    else:
                        st.info("No strong buy signals.")

            with col2:
                with st.container(border=True):
                    st.markdown("🔴 **Top Losers**")
                    sell_df = get_sell_recommendations(df, top_n=5)
                    if not sell_df.empty:
                        st.dataframe(
                            sell_df[["Symbol", "Close", "Change %"]].style.format({
                                "Close": "₹{:.2f}",
                                "Change %": "{:.2f}%"
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No significant losers.")
else:
        
        # ===================================================================
        # Stock-Specific Analysis
        # ===================================================================
        selected_row = df[df["Symbol"] == selected_stock].iloc[0]
        st.markdown(f"#### 📈 **{selected_stock} — Today's Summary**")
        
        # Metrics Row
        hist_df = fetch_stock_history(selected_stock, period=period, interval=timeframe)
        hist_df = add_technical_indicators(hist_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Open", f"₹{selected_row['Open']:.2f}")
        col2.metric("High", f"₹{selected_row['High']:.2f}")
        col3.metric("Low", f"₹{selected_row['Low']:.2f}")
        col4.metric("Close", f"₹{selected_row['Close']:.2f}", f"{selected_row['Change %']:.2f}%")
        col5.metric("Volume", f"{selected_row['Volume']:,}")
        
        #---------------------------------
        st.markdown('----')
        if not hist_df.empty and 'RSI' in hist_df.columns and 'BB_High' in hist_df.columns:
            latest = hist_df.iloc[-1]
            
            st.markdown(f"#### 📊 **{selected_stock} — Technical Indicators**")
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
            col1.metric("RSI (14)", f"{latest['RSI']:.2f}", delta=latest['RSI_Signal'], delta_color="off")
            col2.metric("MACD", f"{latest['MACD']:.3f}")
            col3.metric("MACD Signal", f"{latest['MACD_Signal']:.3f}")
            col4.metric("Histogram", f"{latest['MACD_Hist']:.4f}")
            
            col5.metric("Bollinger Bands - Upper Band", f"₹{latest['BB_High']:.2f}")
            col6.metric("Bollinger Bands - Middle", f"₹{latest['BB_Middle']:.2f}")
            col7.metric("Bollinger Bands - Lower Band", f"₹{latest['BB_Low']:.2f}")
            col8.metric("Bollinger Bands - Position", latest['BB_Position'], delta_color="off")

            st.caption(f"**MACD Crossover Signal**: `{latest['MACD_Signal_Cross']}`")    
                    
        #---------------------------------
        st.markdown('----')
        st.markdown(f"#### 🕯️ **{selected_stock} — {timeframe} Chart**")
        with st.container(border=True):
            fig = sp.make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                subplot_titles=('Price', 'RSI (14)', 'MACD'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # --- Candlestick ---
            fig.add_trace(
                go.Candlestick(
                    x=hist_df['Date'],
                    open=hist_df['Open'],
                    high=hist_df['High'],
                    low=hist_df['Low'],
                    close=hist_df['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # --- Bollinger Bands (only if columns exist) ---
            if 'BB_High' in hist_df.columns and 'BB_Low' in hist_df.columns and 'BB_Middle' in hist_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['BB_High'],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dot'),
                        name='BB Upper',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['BB_Low'],
                        mode='lines',
                        line=dict(color='green', width=1, dash='dot'),
                        name='BB Lower',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['BB_Middle'],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        name='BB Middle',
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # --- RSI ---
            if 'RSI' in hist_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2),
                        showlegend=True
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # --- MACD ---
            if 'MACD' in hist_df.columns and 'MACD_Signal' in hist_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['MACD'],
                        name='MACD',
                        line=dict(color='blue', width=2),
                        showlegend=True
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['MACD_Signal'],
                        name='Signal',
                        line=dict(color='orange', width=2),
                        showlegend=True
                    ),
                    row=3, col=1
                )
                if 'MACD_Hist' in hist_df.columns:
                    fig.add_bar(
                        x=hist_df['Date'],
                        y=hist_df['MACD_Hist'],
                        name='Histogram',
                        marker_color=hist_df['MACD_Hist'].apply(lambda x: 'green' if x >= 0 else 'red'),
                        showlegend=True,
                        row=3, col=1
                    )

            # --- Layout ---
            fig.update_layout(
                height=700,
                template="plotly_white",
                showlegend=True,  # ✅ Now shows BB, RSI, MACD in legend
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)

        #---------------------------------
        st.markdown('----')
        st.markdown("#### 🤖 **ML Prediction**")
        if timeframe == "1d":
            with st.spinner("Generating prediction..."):
                pred_result, error = train_and_predict_next_close(hist_df)
            if error:
                st.error(error)
            else:
                p = pred_result
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Close", f"₹{p['Current_Close']:.2f}")
                col2.metric("Predicted", f"₹{p['Predicted_Next_Close']:.2f}")
                col3.metric("Expected Δ%", f"{p['Expected_Change_%']:.2f}%", delta=f"{p['Expected_Change_%']:.2f}%")
                col4.metric("Signal", p['Signal'],
                            delta_color="normal" if p['Signal'] == "HOLD" else "inverse" if p['Signal'] == "SELL" else "normal")
        else:
            st.info("⚠️ ML prediction is only available for **Daily (1d)** timeframe.")

        #---------------------------------
        
        
        #---------------------------------
        st.markdown('----')
        st.markdown("#### 🚨 **Recent Signals**")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("🟢 **RSI Oversold**")
                oversold = hist_df[hist_df['RSI_Signal'] == 'OVERSOLD'].tail(3)
                if not oversold.empty:
                    st.dataframe(oversold[['Date', 'Close', 'RSI']].style.format({
                        'Close': '₹{:.2f}', 'RSI': '{:.2f}'
                    }), use_container_width=True)
                else:
                    st.info("None recently")
        with col2:
            with st.container(border=True):
                st.markdown("🔴 **MACD Sell Crossovers**")
                macd_sell = hist_df[hist_df['MACD_Signal_Cross'] == 'SELL'].tail(3)
                if not macd_sell.empty:
                    st.dataframe(macd_sell[['Date', 'Close', 'MACD', 'MACD_Signal']].style.format({
                        'Close': '₹{:.2f}', 'MACD': '{:.3f}', 'MACD_Signal': '{:.3f}'
                    }), use_container_width=True)
                else:
                    st.info("None recently")

        #---------------------------------
        st.markdown('----')
        
        with st.expander("Download | Alerts", expanded=True):
        
            col1, col2= st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Stock Data (CSV)",
                    data=csv,
                    file_name="nse_stock_data.csv",
                    mime="text/csv",
                    use_container_width=True)
                
            with col2:
                alert_price = st.number_input("Alert when price reaches (₹)", value=selected_row['Close'], min_value=0.0, step=10.0)
  
                if st.button("🔔 Set Alert"):
                    # Simulate email (replace with smtplib later)
                    st.success(f"✅ Alert set! You'll be notified at ₹{alert_price:.2f} via email.")
                    # In real app: send_email(user_email, f"Alert: {selected_stock} hit ₹{alert_price}")
            
                def send_email(to, subject, body):
                    # Configure your SMTP (Gmail, etc.)
                    pass
                
# ===================================================================
# Footer
# ===================================================================
st.markdown("""
<div class="footer">
    © 2025 | Created by <strong>Avijit Chakraborty</strong> | 
    <a href="mailto:avijit.mba18@gmail.com">📩</a> | 
    <span>Unauthorized use prohibited</span>
</div>
""", unsafe_allow_html=True)
