import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# New Imports for Prophet Forecasting
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# --- Data Loading and Preparation Functions (reusable) ---
@st.cache_data # Still useful if dashboard script is run directly
def load_data():
    """Loads, cleans, and preprocesses the SEBI portfolio data."""
    try:
        df_loaded = pd.read_csv('data/sebi_portfolio_data_complete.csv')
    except FileNotFoundError:
        # For API usage, direct st.error is not ideal. Consider logging or raising exception.
        # For now, if imported, this error won't show in API context unless this func is called by Streamlit.
        if 'streamlit' in globals() and hasattr(st, 'error'): # Check if running in Streamlit context
            st.error("Error: 'data/sebi_portfolio_data_complete.csv' not found.")
        else:
            print("Error: 'data/sebi_portfolio_data_complete.csv' not found.") # For non-Streamlit context
        return pd.DataFrame()

    numeric_cols = df_loaded.select_dtypes(include=['number']).columns
    df_loaded[numeric_cols] = df_loaded[numeric_cols].fillna(0)

    df_loaded['Period'] = pd.to_datetime(
        df_loaded['Year'].astype(str) + '-' + df_loaded['Month'].astype(str) + '-01',
        errors='coerce'
    )
    df_loaded.dropna(subset=['Period'], inplace=True)
    df_loaded = df_loaded.sort_values(by='Period')
    return df_loaded

@st.cache_data # Still useful
def get_latest_data_per_manager(_df):
    """Returns a DataFrame containing only the latest entry for each portfolio manager."""
    if _df.empty or 'Portfolio_Manager' not in _df.columns or 'Period' not in _df.columns:
        return pd.DataFrame()
    latest_idx = _df.groupby('Portfolio_Manager')['Period'].idxmax()
    latest_df = _df.loc[latest_idx]
    return latest_df

# --- New Data Getter Functions (for API, returning JSON-serializable data) ---

def get_total_aum_growth_data(df_input, start_year, end_year):
    """Returns time series data for total AUM growth."""
    df_filtered = df_input[(df_input['Year'] >= start_year) & (df_input['Year'] <= end_year)]
    if df_filtered.empty:
        return []
    total_aum_over_time = df_filtered.groupby('Period')['Total_AUM'].sum().reset_index()
    total_aum_over_time['Period'] = total_aum_over_time['Period'].dt.strftime('%Y-%m-%d')
    return total_aum_over_time.to_dict(orient='records')

def get_top_n_managers_aum_data(df_latest_input, n=10):
    """Returns top N portfolio managers by AUM."""
    if df_latest_input.empty:
        return []
    top_n = df_latest_input.nlargest(n, 'Total_AUM')
    return top_n[['Portfolio_Manager', 'Total_AUM']].to_dict(orient='records')

def get_manager_client_distribution_data(df_latest_input, manager_name):
    """Returns client distribution for a specific manager."""
    if df_latest_input.empty: return None
    manager_data = df_latest_input[df_latest_input['Portfolio_Manager'] == manager_name]
    if manager_data.empty: return None

    client_cols = ['PF/EPFO_Clients', 'Corporates_Clients', 'Non-Corporates_Clients',
                   'Non-Residents_Clients', 'FPI_Clients', 'Others_Clients']
    client_cols_present = [col for col in client_cols if col in manager_data.columns]
    if not client_cols_present: return None

    distribution = manager_data[client_cols_present].iloc[0].to_dict()
    # Filter out zero client categories from the dictionary
    distribution_filtered = {k: v for k, v in distribution.items() if v > 0}

    if not distribution_filtered: return {'manager': manager_name, 'client_distribution': {}}

    return {'manager': manager_name, 'client_distribution': distribution_filtered}

def get_market_share_aum_data(df_latest_input, n=5):
    """Returns market share of top N managers by AUM."""
    if df_latest_input.empty: return []
    total_market_aum = df_latest_input['Total_AUM'].sum()
    if total_market_aum <= 0: return []

    market_share_df = df_latest_input[['Portfolio_Manager', 'Total_AUM']].copy()
    market_share_df['Share'] = (market_share_df['Total_AUM'] / total_market_aum) * 100
    market_share_df = market_share_df.sort_values('Share', ascending=False)

    top_n_market_share = market_share_df.head(n)

    results = top_n_market_share[['Portfolio_Manager', 'Share']].rename(
        columns={'Portfolio_Manager': 'manager', 'Share': 'market_share'}
    ).to_dict(orient='records')

    others_share = market_share_df['Share'][n:].sum()
    if others_share > 0.01: # Only add 'Others' if significant
        results.append({'manager': 'Others', 'market_share': others_share})
    return results

# --- Plotting Functions (for Streamlit Dashboard, unchanged for now but could be refactored) ---
# These functions will be called by run_streamlit_app()
def plot_total_aum_growth(df_input, start_year, end_year):
    df_filtered = df_input[(df_input['Year'] >= start_year) & (df_input['Year'] <= end_year)]
    if df_filtered.empty:
        fig = plt.figure(figsize=(12, 6)); plt.text(0.5, 0.5, 'No data', ha='center'); return fig
    total_aum_over_time = df_filtered.groupby('Period')['Total_AUM'].sum()
    fig = plt.figure(figsize=(12, 6)); sns.lineplot(data=total_aum_over_time)
    plt.title('Total AUM Across All Managers Over Time'); plt.xlabel('Period'); plt.ylabel('Total AUM')
    plt.xticks(rotation=45); plt.tight_layout(); return fig

def plot_top_managers_aum(df_latest_input, N=10):
    if df_latest_input.empty:
        fig = plt.figure(figsize=(12, 8)); plt.text(0.5,0.5, 'No data', ha='center'); return fig
    top_n_managers = df_latest_input.nlargest(N, 'Total_AUM')
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='Total_AUM', y='Portfolio_Manager', data=top_n_managers, palette='viridis', hue='Portfolio_Manager', dodge=False, legend=False)
    plt.title(f'Top {N} Portfolio Managers by Latest Total AUM'); plt.xlabel('Total AUM'); plt.ylabel('Portfolio Manager')
    plt.tight_layout(); return fig

def plot_client_distribution(df_latest_input, manager_name):
    if df_latest_input.empty: return None
    manager_data_dict = get_manager_client_distribution_data(df_latest_input, manager_name) # Use getter
    if not manager_data_dict or not manager_data_dict['client_distribution']: return None

    client_distribution_series = pd.Series(manager_data_dict['client_distribution'])
    fig = plt.figure(figsize=(10, 10))
    plt.pie(client_distribution_series, labels=client_distribution_series.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Client Distribution for {manager_name} (Latest Data)'); plt.axis('equal'); plt.tight_layout(); return fig

def plot_market_share_aum(df_latest_input, N=5):
    market_share_list = get_market_share_aum_data(df_latest_input, N) # Use getter
    if not market_share_list:
        fig = plt.figure(figsize=(10,10)); plt.text(0.5,0.5, 'No data', ha='center'); return fig

    plot_df = pd.DataFrame(market_share_list)
    fig = plt.figure(figsize=(10, 10))
    plt.pie(plot_df['market_share'], labels=plot_df['manager'], autopct='%1.1f%%', startangle=140)
    plt.title(f'Market Share by AUM (Top {N} Managers & Others) - Based on Latest Data')
    plt.axis('equal'); plt.tight_layout(); return fig

@st.cache_data(ttl=3600)
def get_aum_forecast_wrapper(_df, manager_name, periods_to_forecast=8): # Renamed to avoid conflict if run in global scope
    """Generates AUM forecast using Prophet. Wrapper for Streamlit caching."""
    # This function's logic for data prep and Prophet model remains the same
    if manager_name == "Overall Market":
        prophet_df = _df.groupby('Period')['Total_AUM'].sum().reset_index()
    else:
        prophet_df = _df[_df['Portfolio_Manager'] == manager_name][['Period', 'Total_AUM']].copy()
    prophet_df.rename(columns={'Period': 'ds', 'Total_AUM': 'y'}, inplace=True)
    prophet_df = prophet_df.sort_values('ds')
    if len(prophet_df) < 12: return None, None, None # Added None for forecast_data

    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False,
                    seasonality_mode='multiplicative', interval_width=0.95)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods_to_forecast, freq='QS')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast) # This is a Plotly fig, not matplotlib
    fig.update_layout(title=f'AUM Forecast for {manager_name}', xaxis_title='Period', yaxis_title='Total AUM')
    return model, fig, forecast

# --- Main Streamlit App Function ---
def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("SEBI Portfolio Management Analysis")

    df_main = load_data()

    if df_main.empty:
        st.error("Data could not be loaded. Dashboard cannot be displayed.")
        return

    df_latest_entries = get_latest_data_per_manager(df_main)

    st.sidebar.header("Filters")
    manager_list = ["All Managers"] + sorted(df_main["Portfolio_Manager"].unique().tolist())
    selected_manager = st.sidebar.selectbox("Select Portfolio Manager (for Client Dist.)", manager_list)

    unique_years = sorted(df_main['Year'].unique())
    start_year_val = unique_years[0] if unique_years else 2020
    end_year_val = unique_years[-1] if unique_years else 2024
    start_year, end_year = st.sidebar.select_slider(
        "Select Year Range (for AUM Growth)",
        options=unique_years if unique_years else [2020, 2024], # Ensure options is not empty
        value=(start_year_val, end_year_val)
    )

    st.header("Market Overview & Manager Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall AUM Growth Trend")
        st.markdown(f"Displaying AUM growth from **{start_year}** to **{end_year}**.")
        fig_total_aum = plot_total_aum_growth(df_main, start_year, end_year)
        st.pyplot(fig_total_aum)

        st.subheader("Top 5 Portfolio Managers Market Share (by Latest AUM)")
        fig_market_share = plot_market_share_aum(df_latest_entries, N=5)
        st.pyplot(fig_market_share)

    with col2:
        st.subheader(f"Top 10 Portfolio Managers by Latest AUM")
        fig_top_aum = plot_top_managers_aum(df_latest_entries, N=10)
        st.pyplot(fig_top_aum)

        if selected_manager != "All Managers":
            st.subheader(f"Client Distribution for {selected_manager} (Latest Data)")
            fig_client_dist = plot_client_distribution(df_latest_entries, selected_manager)
            if fig_client_dist:
                st.pyplot(fig_client_dist)
            else:
                st.write(f"No client data (>0) to display or not applicable for {selected_manager}.")
        else:
            st.info("Select a specific portfolio manager from the sidebar to view their client distribution.")

    st.header("AUM Forecasting (Next 2 Years, Quarterly)")
    forecast_manager_list = ["Overall Market"] + sorted(df_main["Portfolio_Manager"].unique().tolist())
    selected_forecast_manager = st.selectbox(
        "Select Manager or 'Overall Market' for Forecast",
        forecast_manager_list,
        key="forecast_manager_select"
    )

    periods_to_forecast = 8

    if st.button(f"Generate Forecast for {selected_forecast_manager}", key="run_forecast_button"):
        with st.spinner(f"Generating forecast for {selected_forecast_manager}..."):
            # Use the wrapper for Streamlit caching
            model, forecast_plotly_fig, forecast_data = get_aum_forecast_wrapper(df_main, selected_forecast_manager, periods_to_forecast)

        if forecast_plotly_fig:
            st.plotly_chart(forecast_plotly_fig, use_container_width=True) # Display Plotly fig
            if model is not None and forecast_data is not None:
                st.subheader("Forecast Data (Last 2 Years - Quarterly)")
                st.dataframe(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_forecast))
        else:
            st.warning(f"Not enough data for {selected_forecast_manager} (needs 12+ quarterly points).")

    if st.checkbox("Show Raw Data (Full Dataset)"):
        st.subheader("Raw Data (Full Dataset)"); st.dataframe(df_main)
    if st.checkbox("Show Latest Entries per Manager Data"):
        st.subheader("Latest Entry per Manager"); st.dataframe(df_latest_entries)

# Entry point for running the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
