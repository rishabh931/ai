import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime

# Configure page
st.set_page_config(page_title="Stock P&L Analyzer", layout="wide")
st.title("📈 Indian Stock P&L Statement Analysis")

# Function to format large numbers
def format_number(num):
    if abs(num) >= 1e7:
        return f"₹{num/1e7:,.1f} Cr"
    elif abs(num) >= 1e5:
        return f"₹{num/1e5:,.1f} L"
    elif abs(num) >= 1000:
        return f"₹{num/1000:,.1f} K"
    return f"₹{num:,.2f}"

# Function to find stock symbol
def find_stock_symbol(query):
    query = query.upper().strip()
    if not query.endswith('.NS'):
        query += '.NS'
    if not re.match(r"^[A-Z0-9.-]{1,20}\.NS$", query):
        return None
    return query

# Function to download financial data
def get_financials(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        if stock.info.get('regularMarketPrice') is None:
            return None, None, "Invalid stock symbol"

        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        financial_data = {}

        if 'Total Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Total Revenue'].head(4).values[::-1]
        elif 'Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Revenue'].head(4).values[::-1]
        else:
            return None, None, "Revenue data not available"

        if 'Operating Income' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Income'].head(4).values[::-1]
        elif 'Operating Profit' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Profit'].head(4).values[::-1]
        else:
            return None, None, "Operating Profit data not available"

        if 'Pretax Income' in income_stmt.index:
            financial_data['PBT'] = income_stmt.loc['Pretax Income'].head(4).values[::-1]
        else:
            return None, None, "PBT data not available"

        if 'Net Income' in income_stmt.index:
            financial_data['PAT'] = income_stmt.loc['Net Income'].head(4).values[::-1]
        else:
            return None, None, "PAT data not available"

        if 'Ordinary Shares Number' in balance_sheet.index:
            shares_outstanding = balance_sheet.loc['Ordinary Shares Number'].head(4).values[::-1]
        elif 'Share Issued' in balance_sheet.index:
            shares_outstanding = balance_sheet.loc['Share Issued'].head(4).values[::-1]
        else:
            shares = stock.info.get('sharesOutstanding')
            if shares:
                shares_outstanding = np.array([shares] * 4)
            else:
                return None, None, "Shares outstanding data not available"

        financial_data['EPS'] = financial_data['PAT'] / (shares_outstanding / 1e6)
        financial_data['OPM %'] = (financial_data['Operating Profit'] / financial_data['Revenue']) * 100

        eps_growth = [0]
        for i in range(1, len(financial_data['EPS'])):
            if financial_data['EPS'][i-1] != 0:
                growth = ((financial_data['EPS'][i] - financial_data['EPS'][i-1]) /
                          abs(financial_data['EPS'][i-1]) * 100)
            else:
                growth = 0
            eps_growth.append(growth)
        financial_data['EPS Growth %'] = eps_growth

        years = income_stmt.columns[:4].strftime('%Y').values[::-1]
        return pd.DataFrame(financial_data, index=years), years, None

    except Exception as e:
        return None, None, f"Error: {str(e)}"

# UI Components
st.subheader("Enter any Indian Stock Symbol")
stock_query = st.text_input("Search Stock (e.g., RELIANCE, INFY, TCS):", placeholder="Type stock name or symbol")

if stock_query:
    ticker_symbol = find_stock_symbol(stock_query)

    if not ticker_symbol:
        st.error("Invalid stock symbol format. Please use valid Indian stock symbols.")
    else:
        if st.button("Analyze P&L"):
            with st.spinner(f"Fetching {stock_query} financial data..."):
                financial_df, years, error = get_financials(ticker_symbol)

            if financial_df is not None and error is None:
                st.success(f"Data retrieved for {stock_query} ({ticker_symbol})")

                st.header("1. Revenue, Operating Profit & OPM% Analysis")
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Financial Metrics")
                    display_df = financial_df.copy()
                    display_df['Revenue'] = display_df['Revenue'].apply(format_number)
                    display_df['Operating Profit'] = display_df['Operating Profit'].apply(format_number)
                    display_df['OPM %'] = display_df['OPM %'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_df[['Revenue', 'Operating Profit', 'OPM %']])

                with col2:
                    st.subheader("Performance Trend")
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    width = 0.35
                    x = np.arange(len(years))
                    ax1.bar(x - width/2, financial_df['Revenue']/1e7, width, label='Revenue (₹ Cr)', color='#1f77b4')
                    ax1.bar(x + width/2, financial_df['Operating Profit']/1e7, width, label='Op Profit (₹ Cr)', color='#2ca02c')
                    ax1.set_ylabel('Amount (₹ Crores)')

                    ax2 = ax1.twinx()
                    ax2.plot(x, financial_df['OPM %'], 'r-o', linewidth=2, label='OPM %')
                    ax2.set_ylabel('OPM %', color='#d62728')
                    ax2.tick_params(axis='y', labelcolor='#d62728')
                    plt.title(f'{stock_query} Revenue & Profitability', fontsize=14, fontweight='bold')
                    plt.xticks(x, years)
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    st.pyplot(fig)

                st.header("2. PBT, PAT & EPS Analysis")
                col3, col4 = st.columns([1, 2])

                with col3:
                    st.subheader("Profitability Metrics")
                    display_df2 = financial_df.copy()
                    display_df2['PBT'] = display_df2['PBT'].apply(format_number)
                    display_df2['PAT'] = display_df2['PAT'].apply(format_number)
                    display_df2['EPS'] = display_df2['EPS'].apply(lambda x: f"₹{x:.1f}")
                    display_df2['EPS Growth %'] = display_df2['EPS Growth %'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_df2[['PBT', 'PAT', 'EPS', 'EPS Growth %']])

                with col4:
                    st.subheader("Profitability Trend")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    ax1.bar(years, financial_df['PAT']/1e7, color='#1f77b4', label='PAT (₹ Cr)')
                    ax1.bar(years, (financial_df['PBT'] - financial_df['PAT'])/1e7,
                            bottom=financial_df['PAT']/1e7, color='#ff7f0e', label='Tax (₹ Cr)')
                    ax1.set_ylabel('Amount (₹ Crores)')
                    ax1.legend()
                    ax2.plot(years, financial_df['EPS'], 'g-o', linewidth=2, label='EPS (₹)')
                    ax3 = ax2.twinx()
                    ax3.bar(years, financial_df['EPS Growth %'], alpha=0.4, color='#9467bd', label='EPS Growth %')
                    ax3.set_ylabel('Growth %', color='#9467bd')
                    ax3.tick_params(axis='y', labelcolor='#9467bd')
                    ax3.axhline(0, color='grey', linestyle='--')
                    ax2.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.error(f"Could not retrieve financial data: {error}")

# ✅ Updated: Popular stocks section
st.markdown("### Popular Indian Stocks")
popular_cols = st.columns(5)
popular_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]

for i, stock in enumerate(popular_stocks):
    with popular_cols[i]:
        if st.button(stock, use_container_width=True):
            st.query_params = {"stock": stock}
            st.experimental_rerun()

# ✅ Updated: Handling query params
query_params = st.query_params
if "stock" in query_params:
    stock_query = query_params["stock"][0]
    st.query_params = {}
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("### About This App")
st.markdown(f"""
- **Data Source**: Yahoo Finance (Updated: {datetime.now().strftime('%d %b %Y')})
- **Supported Stocks**: All NSE-listed companies
""")
