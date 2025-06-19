import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # New import
import os # Keep for now, may not be needed

# --- 1. Load Data Function ---
@st.cache_data # Streamlit caching decorator
def load_data():
    """Loads, cleans, and preprocesses the SEBI portfolio data."""
    try:
        df_loaded = pd.read_csv('data/sebi_portfolio_data_complete.csv')
    except FileNotFoundError:
        st.error("Error: 'data/sebi_portfolio_data_complete.csv' not found. Make sure the file path is correct.")
        return pd.DataFrame() # Return empty DataFrame on error

    # Fill missing numeric values with 0
    numeric_cols = df_loaded.select_dtypes(include=['number']).columns
    df_loaded[numeric_cols] = df_loaded[numeric_cols].fillna(0)

    # Convert Year and Month to a single 'Period' datetime object for time series
    # Ensure Year and Month are sorted before creating period for groupby operations
    # This helps if data isn't perfectly sorted in CSV
    df_loaded = df_loaded.sort_values(['Year', 'Month'])
    df_loaded['Period'] = pd.to_datetime(df_loaded['Year'].astype(str) + '-' + df_loaded['Month'].astype(str) + '-01')

    return df_loaded

# --- Visualization Functions (Modified to return figures) ---

def plot_total_aum_growth(df_input, start_year, end_year):
    """Generates and returns the total AUM growth line chart figure."""
    # Filter data based on selected year range
    df_filtered = df_input[(df_input['Year'] >= start_year) & (df_input['Year'] <= end_year)]

    if df_filtered.empty:
        st.warning(f"No data available for the selected year range: {start_year}-{end_year}.")
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'No data available for selected range', ha='center', va='center')
        plt.title('Total AUM Across All Managers Over Time')
        plt.xlabel('Period (Year-Month)')
        plt.ylabel('Total AUM (in Crores)')
        return fig

    total_aum_over_time = df_filtered.groupby('Period')['Total_AUM'].sum()

    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=total_aum_over_time)
    plt.title('Total AUM Across All Managers Over Time')
    plt.xlabel('Period (Year-Month)')
    plt.ylabel('Total AUM (in Crores)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_top_managers_aum(df_input, N=10):
    """Generates and returns the top N portfolio managers by AUM bar chart figure."""
    latest_data_per_manager = df_input.sort_values(['Portfolio_Manager', 'Year', 'Month']).drop_duplicates('Portfolio_Manager', keep='last')
    top_n_managers = latest_data_per_manager.nlargest(N, 'Total_AUM')

    if top_n_managers.empty:
        st.warning(f"No data available to plot top {N} managers.")
        fig = plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f'No data to plot top {N} managers', ha='center', va='center')
        plt.title(f'Top {N} Portfolio Managers by Latest Total AUM')
        return fig

    fig = plt.figure(figsize=(12, 8))
    # Seaborn warning for future: assign y to hue and set legend=False
    sns.barplot(x='Total_AUM', y='Portfolio_Manager', data=top_n_managers, palette='viridis', hue='Portfolio_Manager', dodge=False, legend=False)
    plt.title(f'Top {N} Portfolio Managers by Latest Total AUM')
    plt.xlabel('Total AUM (in Crores)')
    plt.ylabel('Portfolio Manager')
    plt.tight_layout()
    return fig

def plot_client_distribution(df_input, manager_name):
    """Generates and returns the client distribution pie chart figure for a specific manager."""
    latest_data_per_manager = df_input.sort_values(['Portfolio_Manager', 'Year', 'Month']).drop_duplicates('Portfolio_Manager', keep='last')
    manager_data = latest_data_per_manager[latest_data_per_manager['Portfolio_Manager'] == manager_name]

    if manager_data.empty:
        # This case should ideally not be hit if manager_name is from df's unique list
        # st.warning(f"No data found for manager: {manager_name}")
        return None

    client_cols = ['PF/EPFO_Clients', 'Corporates_Clients', 'Non-Corporates_Clients',
                   'Non-Residents_Clients', 'FPI_Clients', 'Others_Clients']
    client_distribution = manager_data[client_cols].iloc[0]
    client_distribution = client_distribution[client_distribution > 0] # Filter out zero client categories

    if client_distribution.empty:
        # st.info(f"No client data (>0) to plot for {manager_name}.")
        return None

    fig = plt.figure(figsize=(10, 10))
    plt.pie(client_distribution, labels=client_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Client Distribution for {manager_name} (Latest Data)')
    plt.axis('equal')
    plt.tight_layout()
    return fig

def plot_market_share_aum(df_input, N=5):
    """Generates and returns the market share pie chart figure for top N managers."""
    latest_data_per_manager = df_input.sort_values(['Portfolio_Manager', 'Year', 'Month']).drop_duplicates('Portfolio_Manager', keep='last')

    if latest_data_per_manager.empty:
        st.warning("No data available for market share analysis.")
        fig = plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, 'No data for market share analysis', ha='center', va='center')
        plt.title(f'Market Share by AUM (Top {N} Managers & Others)')
        return fig

    total_market_aum = latest_data_per_manager['Total_AUM'].sum()

    if total_market_aum <= 0:
        st.warning("Total market AUM is zero or negative. Cannot compute market share.")
        fig = plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, 'Total AUM is zero or negative', ha='center', va='center')
        plt.title(f'Market Share by AUM (Top {N} Managers & Others)')
        return fig

    market_share = latest_data_per_manager[['Portfolio_Manager', 'Total_AUM']].copy()
    market_share['Share'] = (market_share['Total_AUM'] / total_market_aum) * 100
    market_share = market_share.sort_values('Share', ascending=False)

    top_n_market_share = market_share.head(N).copy()
    others_share = market_share['Share'][N:].sum()

    if others_share > 0.01: # Only add 'Others' if significant
        others_df = pd.DataFrame([{'Portfolio_Manager': 'Others', 'Share': others_share}])
        plot_data_market_share = pd.concat([top_n_market_share, others_df], ignore_index=True)
    else:
        plot_data_market_share = top_n_market_share

    if plot_data_market_share.empty or plot_data_market_share['Share'].sum() == 0 :
        st.warning(f"Not enough data to plot market share for top {N} managers.")
        fig = plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, f'Not enough data for market share plot', ha='center', va='center')
        plt.title(f'Market Share by AUM (Top {N} Managers & Others)')
        return fig


    fig = plt.figure(figsize=(10, 10))
    plt.pie(plot_data_market_share['Share'], labels=plot_data_market_share['Portfolio_Manager'], autopct='%1.1f%%', startangle=140)
    plt.title(f'Market Share by AUM (Top {N} Managers & Others) - Based on Latest Data')
    plt.axis('equal')
    plt.tight_layout()
    return fig

# --- Main Streamlit App ---
st.title("SEBI Portfolio Management Analysis")

# Load data using the cached function
df = load_data()

if not df.empty:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Portfolio Manager Selection
    manager_list = ["All Managers"] + sorted(df["Portfolio_Manager"].unique().tolist())
    selected_manager = st.sidebar.selectbox("Select Portfolio Manager", manager_list)

    # Year Range Selection
    unique_years = sorted(df['Year'].unique())
    start_year, end_year = st.sidebar.select_slider(
        "Select Year Range",
        options=unique_years,
        value=(unique_years[0], unique_years[-1]) # Default to full range
    )

    # --- Main Dashboard Area ---

    # 1. Overall AUM Trend
    st.subheader("Overall AUM Growth Trend")
    st.markdown(f"Displaying AUM growth from **{start_year}** to **{end_year}**.")
    fig_total_aum = plot_total_aum_growth(df, start_year, end_year)
    st.pyplot(fig_total_aum)

    # 2. Top Portfolio Managers
    st.subheader(f"Top 10 Portfolio Managers by Latest AUM")
    # Filter df for top managers based on overall latest data, not year slider
    # The plot_top_managers_aum function already uses the latest data for each manager irrespective of year slider
    fig_top_aum = plot_top_managers_aum(df.copy(), N=10) # Pass a copy to avoid potential side-effects if any
    st.pyplot(fig_top_aum)

    # 3. Client Distribution (Conditional Display)
    if selected_manager != "All Managers":
        st.subheader(f"Client Distribution for {selected_manager} (Latest Data)")
        # The plot_client_distribution function also uses latest data for the specific manager
        fig_client_dist = plot_client_distribution(df.copy(), selected_manager)
        if fig_client_dist:
            st.pyplot(fig_client_dist)
        else:
            st.write(f"No client data (>0) to display or not applicable for {selected_manager}.")
    else:
        st.info("Select a specific portfolio manager from the sidebar to view their client distribution.")

    # 4. Market Share
    st.subheader("Top 5 Portfolio Managers Market Share (by Latest AUM)")
    # The plot_market_share_aum function also uses latest data
    fig_market_share = plot_market_share_aum(df.copy(), N=5)
    st.pyplot(fig_market_share)

    # Display some raw data if needed (optional)
    if st.checkbox("Show Raw Data (Latest Entries per Manager)"):
        st.subheader("Raw Data - Latest Entry per Manager")
        latest_entries_df = df.sort_values(['Portfolio_Manager', 'Year', 'Month']).drop_duplicates('Portfolio_Manager', keep='last')
        st.dataframe(latest_entries_df)
else:
    st.warning("Data could not be loaded. Dashboard cannot be displayed.")

# Remove old print statements from previous script version if any were left.
# No need for PLOT_DIR or os.makedirs or explicit plt.savefig or plt.close
# Streamlit handles figure display with st.pyplot()
