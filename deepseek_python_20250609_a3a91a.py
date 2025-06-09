import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date, timedelta
from nsepy import get_history
import openai  # Remove if not using AI analysis

# Configuration
st.set_page_config(page_title="Stock P&L Analyzer", layout="wide")
st.title("📊 Indian Stock P&L Statement Analysis")

# Helper Functions
def get_stock_data(symbol, years=5):
    try:
        # Try fetching with NSE
        end_date = date.today()
        start_date = end_date - timedelta(days=years*365)
        stock = get_history(symbol=symbol, start=start_date, end=end_date)
        if not stock.empty:
            return stock, symbol
    except:
        # Fallback to Yahoo Finance
        ticker = symbol + ".NS"
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{years}y")
        return hist, ticker
    return None, None

def get_financials(ticker):
    stock = yf.Ticker(ticker)
    return stock.financials, stock.balance_sheet

def analyze_with_ai(data, metric):
    """Optional: Uses OpenAI for analysis (requires API key)"""
    # Remove this function if not using OpenAI
    openai.api_key = st.secrets["OPENAI_KEY"]  # Set in Streamlit secrets
    prompt = f"Analyze this {metric} data: {data.to_dict()}. Provide 3 bullet points of insights."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Input Section
symbol = st.text_input("🔍 Enter Indian Stock Symbol (e.g., TCS, INFY, RELIANCE):", "TCS")

if symbol:
    data, ticker = get_stock_data(symbol)
    if data is None:
        st.error("❌ Stock not found! Try symbols like INFY, RELIANCE, HDFCBANK")
        st.stop()

    # Get financial statements
    financials, balance_sheet = get_financials(ticker)
    financials = financials.drop_duplicates().T

    # Filter last 5 years
    financials = financials.head(5).iloc[::-1]  # Reverse for chronological order
    
    # Calculate metrics
    financials['Revenue'] = financials['Total Revenue']
    financials['Operating Profit'] = financials['Operating Income']
    financials['OPM %'] = (financials['Operating Profit'] / financials['Revenue']) * 100
    financials['PBT'] = financials['Pretax Income']
    financials['PAT'] = financials['Net Income']
    financials['EPS'] = financials['Net Income'] / balance_sheet.loc['Ordinary Shares Number'].iloc[0]
    financials['EPS Growth %'] = financials['EPS'].pct_change() * 100

    # Tab Layout
    tab1, tab2 = st.tabs(["Revenue & Operating Profit", "PBT, PAT & EPS"])

    with tab1:
        st.subheader("💰 Revenue, Operating Profit & OPM%")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                financials, 
                x=financials.index.year, 
                y=['Revenue', 'Operating Profit'],
                barmode='group',
                title="Revenue vs Operating Profit"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.line(
                financials, 
                x=financials.index.year, 
                y='OPM %',
                markers=True,
                title="Operating Profit Margin (OPM %)",
                line_shape="linear"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # AI Analysis (Optional)
        st.subheader("🤖 AI-Driven Analysis")
        analysis_points = [
            "• Revenue grew at ₹{0:,.1f} Cr CAGR over 5 years".format(np.mean(financials['Revenue'].pct_change())),
            "• OPM % peaked at {0:.1f}% in {1}".format(
                financials['OPM %'].max(), 
                financials['OPM %'].idxmax().year
            ),
            "• Operating leverage is {} with revenue growth".format(
                "positive" if financials['OPM %'].diff().mean() > 0 else "negative"
            )
        ]
        for point in analysis_points:
            st.info(point)

    with tab2:
        st.subheader("📈 PBT, PAT & EPS Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.area(
                financials, 
                x=financials.index.year, 
                y=['PBT', 'PAT'],
                title="Profit Before Tax (PBT) vs Profit After Tax (PAT)"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
        with col2:
            fig4 = px.bar(
                financials, 
                x=financials.index.year, 
                y='EPS Growth %',
                title="EPS Growth Percentage",
                color=np.where(financials['EPS Growth %'] > 0, 'green', 'red')
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("🤖 AI-Driven Analysis")
        analysis_points = [
            "• PAT margins averaged {0:.1f}% of revenue".format(
                (financials['PAT']/financials['Revenue']).mean()*100
            ),
            "• EPS grew strongest in {0} ({1:.1f}%)".format(
                financials['EPS Growth %'].idxmax().year,
                financials['EPS Growth %'].max()
            ),
            "• Tax efficiency ratio: {0:.1f}%".format(
                (1 - financials['PAT']/financials['PBT']).mean()*100
            )
        ]
        for point in analysis_points:
            st.info(point)

    # Raw Data
    st.subheader("📁 Raw Financial Data")
    st.dataframe(financials[['Revenue', 'Operating Profit', 'OPM %', 'PBT', 'PAT', 'EPS', 'EPS Growth %']])