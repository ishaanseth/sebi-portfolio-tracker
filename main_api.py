from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional, Any
import pandas as pd # For type hinting and potential use, though getters should return lists/dicts

# Import functions from the refactored data_analysis_dashboard
# These functions are designed to return JSON-serializable data or DataFrames
from data_analysis_dashboard import (
    load_data as dashboard_load_data, # Rename to avoid potential global namespace clash if any
    get_latest_data_per_manager as dashboard_get_latest_data,
    get_total_aum_growth_data,
    get_top_n_managers_aum_data,
    get_manager_client_distribution_data,
    get_market_share_aum_data,
    get_aum_forecast_wrapper # For Prophet forecasting, returns model, fig, data
)

app = FastAPI(
    title="SEBI Portfolio Data API",
    description="API for accessing SEBI portfolio management data insights and forecasts.",
    version="1.0.0"
)

# --- Global Data Loading (on API startup) ---
# These calls use the functions from the dashboard script.
# The @st.cache_data in the dashboard script won't apply here unless Streamlit runs it.
# For a production API, consider a separate caching mechanism if needed (e.g., Redis, or FastAPI's own caching utils)
# For this task, direct loading is fine.
main_df_global: pd.DataFrame = dashboard_load_data()
latest_df_global: pd.DataFrame = dashboard_get_latest_data(main_df_global)

# --- API Endpoints ---

@app.get("/", summary="API Root", include_in_schema=False)
async def root():
    return {"message": "Welcome to the SEBI Portfolio Data API. See /docs for available endpoints."}

@app.get("/data/all_cleaned_sample", summary="Get a sample of the cleaned SEBI portfolio data", response_model=List[Dict[str, Any]])
async def get_all_data_sample(limit: Optional[int] = 10):
    if main_df_global.empty:
        raise HTTPException(status_code=404, detail="Data not loaded or empty.")
    return main_df_global.head(limit).to_dict(orient="records")

@app.get("/data/latest_entry/{manager_name}", summary="Get latest AUM and client data for a specific manager", response_model=Dict[str, Any])
async def get_latest_manager_data_api(manager_name: str):
    if latest_df_global.empty:
        raise HTTPException(status_code=404, detail="Latest data not loaded or empty.")

    manager_data_df = latest_df_global[latest_df_global['Portfolio_Manager'] == manager_name]
    if manager_data_df.empty:
        raise HTTPException(status_code=404, detail=f"Manager '{manager_name}' not found in the latest records.")

    # Convert Period to string if it's a datetime object
    result = manager_data_df.to_dict(orient="records")[0]
    if 'Period' in result and hasattr(result['Period'], 'strftime'):
        result['Period'] = result['Period'].strftime('%Y-%m-%d')

    return result

@app.get("/metrics/total_aum_growth", summary="Get time series data for total AUM growth", response_model=List[Dict[str, Any]])
async def get_aum_growth_api(start_year: Optional[int] = None, end_year: Optional[int] = None):
    if main_df_global.empty:
        raise HTTPException(status_code=404, detail="Data not loaded or empty.")

    min_yr_data = main_df_global['Year'].min()
    max_yr_data = main_df_global['Year'].max()

    s_year = start_year if start_year is not None else min_yr_data
    e_year = end_year if end_year is not None else max_yr_data

    if pd.isna(s_year) or pd.isna(e_year): # Handle case where main_df_global might be empty initially
         raise HTTPException(status_code=404, detail="Could not determine year range from data.")

    return get_total_aum_growth_data(main_df_global, int(s_year), int(e_year))

@app.get("/metrics/top_managers_aum", summary="Get top N portfolio managers by latest AUM", response_model=List[Dict[str, Any]])
async def get_top_managers_api(n: Optional[int] = 10):
    if latest_df_global.empty:
        raise HTTPException(status_code=404, detail="Latest data not loaded or empty.")
    return get_top_n_managers_aum_data(latest_df_global, n)

@app.get("/metrics/client_distribution/{manager_name}", summary="Get client distribution for a specific manager", response_model=Dict[str, Any])
async def get_client_distribution_api(manager_name: str):
    if latest_df_global.empty:
        raise HTTPException(status_code=404, detail="Latest data not loaded or empty.")
    data = get_manager_client_distribution_data(latest_df_global, manager_name)
    if not data:
        raise HTTPException(status_code=404, detail=f"Manager '{manager_name}' not found or no client data.")
    return data

@app.get("/metrics/market_share_aum", summary="Get market share of top N managers by latest AUM", response_model=List[Dict[str, Any]])
async def get_market_share_api(n: Optional[int] = 5):
    if latest_df_global.empty:
        raise HTTPException(status_code=404, detail="Latest data not loaded or empty.")
    return get_market_share_aum_data(latest_df_global, n)

@app.get("/forecast/aum/{manager_or_market}", summary="Get AUM forecast for a specific manager or overall market", response_model=Dict[str, Any])
async def get_aum_forecast_api(manager_or_market: str, periods: Optional[int] = 8):
    if main_df_global.empty:
        raise HTTPException(status_code=404, detail="Main data not loaded or empty, cannot generate forecast.")

    # Note: get_aum_forecast_wrapper returns (model, plotly_fig, forecast_df)
    # The plotly_fig is a Plotly JSON object, which FastAPI can serialize.
    # The model itself is not directly serializable.
    _model, fig_json_obj, forecast_df = get_aum_forecast_wrapper(main_df_global, manager_or_market, periods)

    if forecast_df is None:
        raise HTTPException(status_code=400, detail=f"Could not generate forecast for '{manager_or_market}'. Insufficient data (requires at least 12 quarterly data points).")

    # Convert 'ds' (datetime) in forecast_df to string for JSON serialization
    forecast_data_serializable = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    forecast_data_serializable['ds'] = forecast_data_serializable['ds'].dt.strftime('%Y-%m-%d')

    return {
        "manager_or_market": manager_or_market,
        "forecast_periods_quarterly": periods,
        "forecast_plot_plotly_json": fig_json_obj.to_json() if fig_json_obj else None, # Send Plotly figure as JSON
        "forecast_data": forecast_data_serializable.to_dict(orient="records")
    }

# To run this API (after installing uvicorn and fastapi):
# uvicorn main_api:app --reload
# Then access docs at http://127.0.0.1:8000/docs
