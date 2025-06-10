# app.py
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
    if abs(num) >= 1e7:  # Crores
        return f"₹{num/1e7:,.1f} Cr"
    elif abs(num) >= 1e5:  # Lakhs
        return f"₹{num/1e5:,.1f} L"
    elif abs(num) >= 1000:  # Thousands
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
        
        # Revenue
        if 'Total Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Total Revenue'].head(4).values[::-1]
        elif 'Revenue' in income_stmt.index:
            financial_data['Revenue'] = income_stmt.loc['Revenue'].head(4).values[::-1]
        else:
            return None, None, "Revenue data not available"
        
        # Operating Profit
        if 'Operating Income' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Income'].head(4).values[::-1]
        elif 'Operating Profit' in income_stmt.index:
            financial_data['Operating Profit'] = income_stmt.loc['Operating Profit'].head(4).values[::-1]
        else:
            return None, None, "Operating Profit data not available"
        
        # PBT
        if 'Pretax Income' in income_stmt.index:
            financial_data['PBT'] = income_stmt.loc['Pretax Income'].head(4).values[::-1]
        else:
            return None, None, "PBT data not available"
        
        # PAT
        if 'Net Income' in income_stmt.index:
            financial_data['PAT'] = income_stmt.loc['Net Income'].head(4).values[::-1]
        else:
            return None, None, "PAT data not available"
        
        # Shares Outstanding
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
        
        # Calculate EPS
        financial_data['EPS'] = financial_data['PAT'] / (shares_outstanding / 1e6)
        
        # Calculate metrics
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
        
        # Get years
        years = income_stmt.columns[:4].strftime('%Y').values[::-1]
        
        return pd.DataFrame(financial_data, index=years), years, None
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# AI-driven analysis functions
def generate_revenue_analysis(df):
    insights = ["### 🔍 Revenue & Profitability Insights"]
    
    # Revenue analysis
    rev_growth = df['Revenue'].pct_change().dropna() * 100
    latest_rev = df['Revenue'].iloc[-1]
    initial_rev = df['Revenue'].iloc[0]
    
    if len(df) > 1:
        cagr = ((latest_rev / initial_rev) ** (1/(len(df)-1)) - 1) * 100
        avg_growth = rev_growth.mean()
        
        # AI-driven insights
        insights.append("- **Revenue Trend Analysis**:")
        if cagr > 20:
            insights.append(f"  - 🚀 **Strong Growth**: Revenue growing at impressive {cagr:.1f}% CAGR")
        elif cagr > 10:
            insights.append(f"  - 📈 **Healthy Expansion**: Steady revenue growth at {cagr:.1f}% CAGR")
        elif cagr > 0:
            insights.append(f"  - ↗️ **Moderate Growth**: Revenue increasing at {cagr:.1f}% CAGR")
        else:
            insights.append(f"  - ⚠️ **Revenue Decline**: Revenue contracting at {abs(cagr):.1f}% CAGR")
            
        insights.append(f"  - Recent momentum: {'Accelerating' if rev_growth.iloc[-1] > avg_growth else 'Decelerating'} growth in latest year")
    
    # OPM analysis
    current_opm = df['OPM %'].iloc[-1]
    initial_opm = df['OPM %'].iloc[0]
    opm_change = current_opm - initial_opm
    
    insights.append("\n- **Margin Analysis**:")
    if opm_change > 5:
        insights.append(f"  - 🏆 **Margin Expansion**: Significant OPM improvement from {initial_opm:.1f}% to {current_opm:.1f}% (+{opm_change:.1f}%)")
    elif opm_change > 1:
        insights.append(f"  - 📊 **Positive Trend**: Moderate OPM improvement from {initial_opm:.1f}% to {current_opm:.1f}%")
    elif opm_change < -3:
        insights.append(f"  - ⚠️ **Margin Pressure**: OPM declined from {initial_opm:.1f}% to {current_opm:.1f}% ({opm_change:.1f}%)")
    else:
        insights.append(f"  - ↔️ **Stable Margins**: OPM range between {min(df['OPM %']):.1f}%-{max(df['OPM %']):.1f}%")
    
    # Profit efficiency
    if len(df) > 1:
        profit_leverage = ((df['Operating Profit'].pct_change().mean() - 
                          df['Revenue'].pct_change().mean())) * 100
        insights.append(f"  - ⚙️ **Operating Leverage**: {profit_leverage:.1f}% (profit growth vs revenue growth)")
    
    return "\n".join(insights)

def generate_profitability_analysis(df):
    insights = ["### 💰 Profitability & EPS Insights"]
    
    # PAT analysis
    pat_growth = df['PAT'].pct_change().dropna() * 100
    latest_pat = df['PAT'].iloc[-1]
    initial_pat = df['PAT'].iloc[0]
    
    if len(df) > 1:
        pat_cagr = ((latest_pat / initial_pat) ** (1/(len(df)-1)) - 1) * 100
        avg_pat_growth = pat_growth.mean()
        
        insights.append("- **Profit Trend Analysis**:")
        if pat_cagr > 25:
            insights.append(f"  - 🚀 **Exceptional Profit Growth**: PAT growing at {pat_cagr:.1f}% CAGR")
        elif pat_cagr > 15:
            insights.append(f"  - 📈 **Strong Profitability**: PAT growth at {pat_cagr:.1f}% CAGR")
        elif pat_cagr > 0:
            insights.append(f"  - ↗️ **Moderate Growth**: PAT increasing at {pat_cagr:.1f}% CAGR")
        else:
            insights.append(f"  - ⚠️ **Profit Decline**: PAT contracting at {abs(pat_cagr):.1f}% CAGR")
        
        # Compare to revenue growth
        rev_cagr = ((df['Revenue'].iloc[-1] / df['Revenue'].iloc[0]) ** (1/(len(df)-1)) - 1) * 100
        if pat_cagr > rev_cagr + 5:
            insights.append(f"  - ⬆️ **Profit Expansion**: PAT growing faster than revenue (+{pat_cagr-rev_cagr:.1f}%)")
        elif pat_cagr < rev_cagr - 5:
            insights.append(f"  - ⬇️ **Margin Pressure**: PAT growing slower than revenue ({pat_cagr-rev_cagr:.1f}% gap)")
    
    # Tax efficiency
    if len(df) > 0:
        tax_efficiency = (df['PAT'] / df['PBT']).mean() * 100
        insights.append(f"- **Tax Efficiency**: Average {tax_efficiency:.1f}% PAT retention from PBT")
    
    # EPS analysis
    latest_eps = df['EPS'].iloc[-1]
    eps_growth = df['EPS Growth %'].dropna()
    
    insights.append("\n- **EPS Analysis**:")
    insights.append(f"  - Current EPS: ₹{latest_eps:.1f}")
    
    if len(eps_growth) > 0:
        avg_eps_growth = eps_growth.mean()
        consistency = "Stable" if eps_growth.std() < 15 else "Volatile"
        
        if avg_eps_growth > 20:
            insights.append(f"  - 🚀 **High Growth**: EPS growing at {avg_eps_growth:.1f}% annually")
        elif avg_eps_growth > 10:
            insights.append(f"  - 📈 **Healthy Growth**: Consistent EPS growth at {avg_eps_growth:.1f}%")
        elif avg_eps_growth > 0:
            insights.append(f"  - ↗️ **Moderate Growth**: EPS increasing at {avg_eps_growth:.1f}%")
        else:
            insights.append(f"  - ⚠️ **EPS Decline**: Negative growth trend ({avg_eps_growth:.1f}%)")
        
        insights.append(f"  - Growth Consistency: {consistency} (yearly changes: {eps_growth.min():.1f}% to {eps_growth.max():.1f}%)")
        
        # Compare to PAT growth
        if len(df) > 1 and avg_pat_growth:
            eps_vs_pat = avg_eps_growth - avg_pat_growth
            if abs(eps_vs_pat) > 5:
                if eps_vs_pat > 0:
                    insights.append(f"  - ⬆️ **Shareholder Advantage**: EPS growth exceeds PAT growth by {eps_vs_pat:.1f}%")
                else:
                    insights.append(f"  - ⬇️ **Share Dilution**: EPS growth lags PAT growth by {abs(eps_vs_pat):.1f}%")
    
    # Valuation insight
    if len(df) > 0:
        pe_ratio = 20  # Placeholder - would require current price
        eps_growth_rate = avg_eps_growth if 'avg_eps_growth' in locals() else 0
        peg_ratio = pe_ratio / eps_growth_rate if eps_growth_rate > 0 else 0
        
        insights.append("\n- **Valuation Insight**:")
        if peg_ratio < 1 and eps_growth_rate > 15:
            insights.append(f"  - 💎 **Undervalued**: PEG ratio of {peg_ratio:.1f} suggests potential undervaluation")
        elif peg_ratio < 1.5:
            insights.append(f"  - ⚖️ **Fair Value**: PEG ratio of {peg_ratio:.1f} indicates reasonable valuation")
        else:
            insights.append(f"  - ⚠️ **Premium Valuation**: PEG ratio of {peg_ratio:.1f} suggests premium pricing")
    
    return "\n".join(insights)

# UI Components
st.subheader("Enter any Indian Stock Symbol")
stock_query = st.text_input("Search Stock (e.g., RELIANCE, INFY, TCS):", 
                           placeholder="Type stock name or symbol")

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
                
                # Section 1: Revenue, Operating Profit, OPM%
                st.header("1. Revenue, Operating Profit & OPM% Analysis")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Financial Metrics")
                    display_df = financial_df.copy()
                    display_df['Revenue'] = display_df['Revenue'].apply(format_number)
                    display_df['Operating Profit'] = display_df['Operating Profit'].apply(format_number)
                    display_df['OPM %'] = display_df['OPM %'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(display_df[['Revenue', 'Operating Profit', 'OPM %']])
                    
                    csv = financial_df[['Revenue', 'Operating Profit', 'OPM %']].to_csv()
                    st.download_button(
                        label="Download Revenue Data",
                        data=csv,
                        file_name=f"{stock_query}_revenue_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.subheader("Performance Trend")
                    
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    width = 0.35
                    x = np.arange(len(years))
                    ax1.bar(x - width/2, financial_df['Revenue']/1e7, width, 
                            label='Revenue (₹ Cr)', color='#1f77b4')
                    ax1.bar(x + width/2, financial_df['Operating Profit']/1e7, width, 
                            label='Op Profit (₹ Cr)', color='#2ca02c')
                    ax1.set_ylabel('Amount (₹ Crores)')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(x, financial_df['OPM %'], 'r-o', linewidth=2, label='OPM %')
                    ax2.set_ylabel('OPM %', color='#d62728')
                    ax2.tick_params(axis='y', labelcolor='#d62728')
                    
                    if max(financial_df['OPM %']) > 0:
                        ax2.set_ylim(0, max(financial_df['OPM %']) * 1.5)
                    
                    plt.title(f'{stock_query} Revenue & Profitability', fontsize=14, fontweight='bold')
                    plt.xticks(x, years)
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                
                # AI-driven insights
                st.subheader("📊 AI-Driven Analysis")
                with st.expander("View Detailed Insights", expanded=True):
                    st.markdown(generate_revenue_analysis(financial_df))
                
                # Section 2: PBT, PAT, EPS
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
                    
                    csv2 = financial_df[['PBT', 'PAT', 'EPS', 'EPS Growth %']].to_csv()
                    st.download_button(
                        label="Download Profitability Data",
                        data=csv2,
                        file_name=f"{stock_query}_profitability_data.csv",
                        mime="text/csv"
                    )
                
                with col4:
                    st.subheader("Profitability Trend")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # PAT and PBT trend
                    ax1.bar(years, financial_df['PAT']/1e7, color='#1f77b4', label='PAT (₹ Cr)')
                    ax1.bar(years, (financial_df['PBT'] - financial_df['PAT'])/1e7, 
                            bottom=financial_df['PAT']/1e7, color='#ff7f0e', label='Tax (₹ Cr)')
                    ax1.set_ylabel('Amount (₹ Crores)')
                    ax1.set_title('Profit Before Tax Composition', fontsize=12, fontweight='bold')
                    ax1.legend()
                    ax1.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # EPS Growth
                    ax2.plot(years, financial_df['EPS'], 'g-o', linewidth=2, label='EPS (₹)')
                    ax2.set_ylabel('EPS', color='#2ca02c')
                    ax2.tick_params(axis='y', labelcolor='#2ca02c')
                    
                    ax3 = ax2.twinx()
                    ax3.bar(years, financial_df['EPS Growth %'], alpha=0.4, color='#9467bd', label='EPS Growth %')
                    ax3.set_ylabel('Growth %', color='#9467bd')
                    ax3.tick_params(axis='y', labelcolor='#9467bd')
                    ax3.axhline(0, color='grey', linestyle='--')
                    ax3.set_title('EPS Performance', fontsize=12, fontweight='bold')
                    ax3.legend(loc='lower right')
                    ax2.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # AI-driven insights for profitability
                st.subheader("📊 AI-Driven Analysis")
                with st.expander("View Detailed Insights", expanded=True):
                    st.markdown(generate_profitability_analysis(financial_df))
                
                # GitHub resources
                st.header("📦 GitHub Resources")
                st.info("All code and data files are available in the GitHub repository:")
                st.markdown("[![GitHub](https://img.shields.io/badge/Repository-100000?logo=github)](https://github.com/yourusername/stock-pl-analysis)")
                
                resources = {
                    "app.py": "Main Streamlit application code",
                    "requirements.txt": "Python dependencies",
                    "stock_data.csv": "Sample dataset",
                    "analysis_template.ipynb": "Jupyter Notebook for analysis"
                }
                
                for file, description in resources.items():
                    with st.expander(f"Download {file}"):
                        st.write(description)
                        content = f"This is a sample {file} file for {stock_query} analysis"
                        st.download_button(
                            label=f"Download {file}",
                            data=content,
                            file_name=file,
                            mime="text/plain"
                        )
            else:
                st.error(f"Could not retrieve financial data: {error}")

# Popular stocks
st.markdown("### Popular Indian Stocks")
popular_cols = st.columns(5)
popular_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]

for i, stock in enumerate(popular_stocks):
    with popular_cols[i]:
        if st.button(stock, use_container_width=True):
            st.experimental_set_query_params(stock=stock)
            st.experimental_rerun()

# Handle query params
query_params = st.experimental_get_query_params()
if "stock" in query_params:
    stock_query = query_params["stock"][0]
    st.experimental_set_query_params()
    st.experimental_rerun()

# Add footer
st.markdown("---")
st.markdown("### About This App")
st.markdown(f"""
- **Data Source**: Yahoo Finance (Updated: {datetime.now().strftime('%d %b %Y')})
- **Analysis Features**: 
  - AI-driven financial insights
  - 4-year trend visualizations
  - Profitability metrics with growth analysis
- **Number Formatting**: 
  - ₹1,000 = ₹1K 
  - ₹100,000 = ₹1L 
  - ₹10,000,000 = ₹1Cr
- **Supported Stocks**: All NSE-listed companies
""")
