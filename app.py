# app.py
#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Template Graphics
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
#----------------------------------------
# Plots
import altair as alt
import plotly.express as px
import plotly.subplots as sp
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#----------------------------------------
import scikitplot as skplt
#from scikitplot.metrics import plot_lift_curve, plot_cumulative_gain
from mpl_toolkits.mplot3d import Axes3D
#---------------------------------------------------------------------------------------------------------------------------------
### Import Functions
#---------------------------------------------------------------------------------------------------------------------------------
from data_fetcher import fetch_nse_data, fetch_stock_history
from ml_predictor import train_and_predict_next_close
from strategies import add_technical_indicators, get_buy_recommendations, get_sell_recommendations
#---------------------------------------------------------------------------------------------------------------------------------
### Title for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="Stock Dashboard | v1.0",
                   layout="wide",
                   page_icon="📈",              
                   initial_sidebar_state="auto")
#---------------------------------------------------------------------------------------------------------------------------------
### CSS
#---------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### Description for your Streamlit app
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

#----------------------------------------
#st.markdown('<div class="centered-info"><span style="margin-left: 10px;">A lightweight Machine Learning (ML) streamlit app that help to analyse different types machine learning problems</span></div>',unsafe_allow_html=True,)
#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------
st.divider()

#----------------------------------------
col1, col2= st.columns((0.15,0.85))
with col1:
    
    if st.button("**🔄 Refresh Data**",use_container_width=True):
        st.cache_data.clear()  # Clear cache to force reload
        st.rerun()           
        
    with st.expander('**🔍 Stock Finder**', expanded=True):
            
        df = fetch_nse_data()
        selected_stock = st.selectbox("Choose a stock", [""] + sorted(df["Symbol"].tolist()))
            
    if selected_stock == "":
        st.divider()
    
    else:
        st.divider()
        with st.expander('**📅 TimeFrame**', expanded=True):
                    timeframe = st.selectbox("Chart Timeframe",options=["1d", "5m", "15m", "1h", "1wk"],index=0)
                    period_map = {"5m": "60d","15m": "60d","1h": "730d","1d": "3mo","1wk": "2y"}
        period = period_map[timeframe]            

with col2:
    
    with st.expander('**🏆 Top Performers Today**', expanded=True):               
                    
        top_gainers = df.head(5)
        st.dataframe(top_gainers.style.format({
                        "Open": "₹{:.2f}",
                        "High": "₹{:.2f}",
                        "Low": "₹{:.2f}",
                        "Close": "₹{:.2f}",
                        "Change %": "{:.2f}%"
                    }))            
    
    with st.expander('**🎯 Smart Recommendations**', expanded=False):     

        buy_candidates = get_buy_recommendations(df, top_n=10)
        ml_buy_list = []

        for symbol in df["Symbol"].head(10):  # Top 10 by performance
            hist = fetch_stock_history(symbol, period="3mo")
            pred, err = train_and_predict_next_close(hist)
            if pred and pred["Signal"] == "BUY":
                    ml_buy_list.append({"Symbol": symbol,"Predicted_Change_%": pred["Expected_Change_%"],"ML_Signal": pred["Signal"]})

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
            
                st.markdown("🟢 ML + Performance (Consider BUY)")
                if ml_buy_list:
                    ml_buy_df = pd.DataFrame(ml_buy_list).sort_values("Predicted_Change_%", ascending=False)
                    st.dataframe(ml_buy_df.style.format({"Predicted_Change_%": "{:.2f}%"}))
                else:
                    st.info("No strong ML BUY signals today.")

            with st.container(border=True):
                
                st.markdown("🟢 RSI Oversold (Potential Buy)")
                hist_df = fetch_stock_history(selected_stock, period=period, interval=timeframe)
                hist_df = add_technical_indicators(hist_df)
                oversold = hist_df[hist_df['RSI_Signal'] == 'OVERSOLD']
                if not oversold.empty:
                    st.dataframe(oversold[['Date', 'Close', 'RSI', 'RSI_Signal']].tail(5).style.format({
                        'Close': '₹{:.2f}',
                        'RSI': '{:.2f}'
                    }))
                else:
                    st.info("No RSI oversold signals recently.")

        with col2:
            with st.container(border=True):
            
                st.markdown("🔴 Top Losers (Consider SELL)")
                sell_df = get_sell_recommendations(df, top_n=5)
                if not sell_df.empty:
                    st.dataframe(sell_df[["Symbol", "Close", "Change %"]].style.format({"Close": "₹{:.2f}","Change %": "{:.2f}%"}))
                else:
                    st.info("No significant losers today.")

            with st.container(border=True):
                
                st.markdown("🔴 MACD Sell Crossovers")
                macd_sell = hist_df[hist_df['MACD_Signal_Cross'] == 'SELL']
                if not macd_sell.empty:
                    st.dataframe(macd_sell[['Date', 'Close', 'MACD', 'MACD_Signal', 'MACD_Signal_Cross']].tail(5).style.format({
                        'Close': '₹{:.2f}',
                        'MACD': '{:.3f}',
                        'MACD_Signal': '{:.3f}'
                    }))
                else:
                    st.info("No recent MACD sell crossovers.")
                
     
    if selected_stock == "":
            st.warning("Please choose a stock to know more.")
            
    else:
            
            st.divider()
                    
            selected_row = df[df["Symbol"] == selected_stock].iloc[0]
            with st.expander(f"**📈 {selected_stock} - Today's Metrics**", expanded=True):

                col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
                col1.metric("Open", f"₹{selected_row['Open']:.2f}")
                col2.metric("High", f"₹{selected_row['High']:.2f}")
                col3.metric("Low", f"₹{selected_row['Low']:.2f}")
                col4.metric("Close", f"₹{selected_row['Close']:.2f}", f"{selected_row['Change %']:.2f}%")
                col5.metric("Volume", f"{selected_row['Volume']:,}")

                hist_df = fetch_stock_history(selected_stock, period=period, interval=timeframe)
                hist_df = add_technical_indicators(hist_df)
                
                if not hist_df.empty and 'RSI' in hist_df.columns:
                    latest = hist_df.iloc[-1]

                    col6.metric("RSI (14)", f"{latest['RSI']:.2f}", delta=latest['RSI_Signal'], delta_color="off")
                    col7.metric("MACD", f"{latest['MACD']:.3f}")
                    col8.metric("MACD Signal", f"{latest['MACD_Signal']:.3f}")

                    macd_hist = latest['MACD_Hist']
                    color = "green" if macd_hist > 0 else "red"
                    col9.metric("MACD Histogram", f"{macd_hist:.4f}", delta=None)
                    col10.write(f"**MACD Crossover Signal**: `{latest['MACD_Signal_Cross']}`")
                    

            with st.expander(f"*🕯️{selected_stock} — {timeframe} Chart with Indicators**", expanded=False):                    
                        
                #fig = go.Figure(data=[go.Candlestick(x=hist_df['Date'],open=hist_df['Open'],high=hist_df['High'],low=hist_df['Low'],close=hist_df['Close'],name=selected_stock)])
                #fig.update_layout(title=f"{selected_stock} Price Movement",xaxis_title="Date",yaxis_title="Price (₹)",xaxis_rangeslider_visible=False,height=500,template="plotly_dark")
                #st.plotly_chart(fig, use_container_width=True)

                # Create subplots: Candlestick + RSI + MACD
                fig = sp.make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price', 'RSI (14)', 'MACD'),
                    row_heights=[0.6, 0.2, 0.2]
                )

                # Candlestick
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
                # RSI
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # MACD
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['MACD'],
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'],
                        y=hist_df['MACD_Signal'],
                        name='MACD Signal',
                        line=dict(color='orange', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_bar(
                    x=hist_df['Date'],
                    y=hist_df['MACD_Hist'],
                    name='MACD Histogram',
                    marker_color=hist_df['MACD_Hist'].apply(lambda x: 'green' if x >= 0 else 'red'),
                    row=3, col=1
                )

                # Update layout
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified"
                )
                fig.update_xaxes(title_text="Date", row=3, col=1)

                st.plotly_chart(fig, use_container_width=True)
                
            with st.expander("**🤖 ML Prediction: Next Day Close**", expanded=True):

                if timeframe == "1d":  # Only run ML on daily data for reliability
                    with st.spinner("Training model..."):
                        pred_result, error = train_and_predict_next_close(hist_df)

                    if error:
                        st.error(f"Prediction failed: {error}")
                    else:
                        p = pred_result
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Current Close", f"₹{p['Current_Close']:.2f}")
                        col2.metric("Predicted Close", f"₹{p['Predicted_Next_Close']:.2f}")
                        col3.metric("Expected Change", f"{p['Expected_Change_%']:.2f}%", delta=f"{p['Expected_Change_%']:.2f}%")
                        col4.metric("ML Signal", p['Signal'], delta=p['Signal'], 
                                    delta_color="normal" if p['Signal'] == "HOLD" else "inverse" if p['Signal'] == "SELL" else "normal")
                else:
                    st.info("ML Prediction is only available for Daily (1d) timeframe.")

            if st.checkbox("Show Raw Data"):
                        st.dataframe(df.style.format({
                            "Open": "₹{:.2f}",
                            "High": "₹{:.2f}",
                            "Low": "₹{:.2f}",
                            "Close": "₹{:.2f}",
                            "Change %": "{:.2f}%"
                        }))

            st.caption("💡 Tip: Use at your own risk. Not financial advice. ML model is baseline Linear Regression.")