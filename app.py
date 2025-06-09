import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# Configuration
st.set_page_config(page_title="Stock P&L Analyzer", layout="wide")
st.title("📊 Indian Stock P&L Statement Analysis")

# Helper function to find correct column name
def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    st.error(f"❌ Could not find financial metric. Tried: {', '.join(possible_names)}")
    st.error(f"Available columns: {', '.join(df.columns)}")
    st.stop()

# Get financial data
def get_financials(symbol):
    try:
        ticker = symbol + ".NS"
        stock = yf.Ticker(ticker)
        financials = stock.financials.T
        balance_sheet = stock.balance_sheet.T
        
        # Handle different column naming conventions
        revenue_col = find_column(financials, ['Total Revenue', 'Revenue'])
        op_col = find_column(financials, ['Operating Income', 'Operating Profit'])
        pbt_col = find_column(financials, ['Pretax Income', 'Income Before Tax'])
        pat_col = find_column(financials, ['Net Income', 'Net Income From Continuing Ops'])
        
        # Get shares outstanding
        shares_col = find_column(balance_sheet, ['Ordinary Shares Number', 'Share Issued'])
        shares = balance_sheet[shares_col].iloc[0]
        
        # Get latest 5 years
        financials = financials.head(5).iloc[::-1]
        
        # Calculate metrics
        financials['Revenue'] = financials[revenue_col]
        financials['Operating Profit'] = financials[op_col]
        financials['OPM %'] = (financials['Operating Profit'] / financials['Revenue']) * 100
        financials['PBT'] = financials[pbt_col]
        financials['PAT'] = financials[pat_col]
        financials['EPS'] = financials['PAT'] / shares
        financials['EPS Growth %'] = financials['EPS'].pct_change() * 100
        
        return financials, None
    
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None, str(e)

# UI Elements
symbol = st.text_input("🔍 Enter Indian Stock Symbol (e.g., TCS, INFY, RELIANCE):", "TCS")

if symbol:
    st.subheader(f"📈 Financial Analysis: {symbol}")
    financials, error = get_financials(symbol)
    
    if financials is not None:
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
            
            # Revenue Analysis
            st.subheader("🔍 Key Insights")
            revenue_growth = financials['Revenue'].pct_change().mean() * 100
            avg_opm = financials['OPM %'].mean()
            
            st.markdown(f"""
            - **Revenue Growth**: {revenue_growth:.1f}% average annual growth
            - **OPM Range**: {financials['OPM %'].min():.1f}% to {financials['OPM %'].max():.1f}%
            - **Profit Stability**: {'Stable' if financials['OPM %'].std() < 5 else 'Volatile'} margins
            - **Efficiency Trend**: {'Improving' if financials['OPM %'].iloc[-1] > financials['OPM %'].iloc[0] else 'Declining'} profitability
            """)

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
            
            # Profitability Analysis
            st.subheader("🔍 Key Insights")
            eps_growth = financials['EPS Growth %'].mean()
            tax_efficiency = (1 - (financials['PAT'] / financials['PBT']).mean() * 100
            
            st.markdown(f"""
            - **EPS Growth**: {eps_growth:.1f}% average annual increase
            - **Tax Efficiency**: {tax_efficiency:.1f}% effective tax rate
            - **Profit Conversion**: {(financials['PAT']/financials['PBT']).mean()*100:.1f}% PBT to PAT conversion
            - **Growth Consistency**: {sum(financials['EPS Growth %'] > 0)} years of positive EPS growth
            """)

        # Raw Data
        st.subheader("📁 Raw Financial Data")
        st.dataframe(financials[['Revenue', 'Operating Profit', 'OPM %', 'PBT', 'PAT', 'EPS', 'EPS Growth %']])
