import pandas as pd
from dash import dcc, html, dash_table, callback_context
import dash.dependencies as ddep
import dash
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta

from app import app, db

# Define expected columns for consistency, even when empty
EXPECTED_COLUMNS = [
    "Cid",
    "Company",
    "Symbol",
    "Date",
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "Mean",
    "CloseValue",
]

def get_all_company_options():
    """
    Fetches all companies for populating the dropdown.
    """
    query = """
        SELECT DISTINCT
            c.id as cid,
            c.name as company,
            c.symbol as symbol
        FROM companies c
        INNER JOIN daystocks ds ON c.id = ds.cid
        ORDER BY c.name ASC
    """
    companies_df = db.df_query(query)

    if companies_df.empty:
        return []

    companies_df["CompanyDisplay"] = (
        companies_df["company"] + " (" + companies_df["symbol"] + ")"
    )

    return [
        {"label": row["CompanyDisplay"], "value": row["cid"]}
        for _, row in companies_df.iterrows()
    ]

def get_company_date_range(cids=None):
    """
    Fetches the overall min and max date for a given list of company Cids.
    """
    if not cids:
        return None, None

    if isinstance(cids, (int, str)):
        cids = [cids]

    query = """
        SELECT MIN(ds.date) as min_date, MAX(ds.date) as max_date
        FROM daystocks ds
        WHERE ds.cid IN %(cids)s
    """
    params = {"cids": tuple(cids)}

    try:
        result_df = db.df_query(query, params=params)
        if not result_df.empty and not pd.isna(result_df.iloc[0]["min_date"]):
            min_date = pd.to_datetime(result_df.iloc[0]["min_date"], utc=True)
            max_date = pd.to_datetime(result_df.iloc[0]["max_date"], utc=True)
            return min_date, max_date
        else:
            return None, None
    except Exception as e:
        print(f"!!! ERROR during get_company_date_range: {e}")
        return None, None

def get_stock_data(cids=None, start_date=None, end_date=None):
    """
    Fetches daily stock data for given Cids within a specific date range.
    """
    print(
        f"--- get_stock_data (tab2) called with cids: {cids}, start: {start_date}, end: {end_date} ---"
    )

    if not cids:
        print("No CIDs provided, returning empty DataFrame.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if isinstance(cids, (int, str)):
        cids = [cids]

    query = """
        SELECT
            c.id as cid,
            c.name as company,
            c.symbol as symbol,
            ds.date as date,
            ds.open as open,
            ds.close as close,
            ds.high as high,
            ds.low as low,
            ds.volume as volume,
            ds.mean as mean,
            ds.close as close_value
        FROM companies c
        JOIN daystocks ds ON c.id = ds.cid
        WHERE c.id IN %(cids)s
    """
    params = {"cids": tuple(cids)}

    if start_date:
        query += " AND ds.date >= %(start_date)s"
        params["start_date"] = start_date
    if end_date:
        end_date_sql = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        query += " AND ds.date < %(end_date)s"
        params["end_date"] = end_date_sql

    query += " ORDER BY ds.date DESC"

    print(f"Executing SQL Query (tab2):\n{query}")
    print(f"With Parameters (tab2): {params}")

    try:
        companies_df = db.df_query(query, params=params)
        print(f"Query returned DataFrame shape (tab2): {companies_df.shape}")
    except Exception as e:
        print(f"!!! ERROR during db.df_query (tab2): {e}")
        companies_df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    if companies_df.empty:
        print("Query returned an empty DataFrame (tab2).")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    companies_df["date"] = pd.to_datetime(companies_df["date"], utc=True)
    companies_df = companies_df.rename(
        columns={
            "cid": "Cid",
            "company": "Company",
            "symbol": "Symbol",
            "date": "Date",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
            "mean": "Mean",
            "close_value": "CloseValue",
        }
    )

    for col in EXPECTED_COLUMNS:
        if col not in companies_df.columns:
            companies_df[col] = None

    print(
        f"--- get_stock_data (tab2) finished, returning shape: {companies_df[EXPECTED_COLUMNS].shape} ---"
    )
    return companies_df[EXPECTED_COLUMNS]


tab2_layout = html.Div(
    [
        html.Link(
            rel="stylesheet",
            href="https://use.fontawesome.com/releases/v5.15.4/css/all.css"
        ),
        html.Div(
            [
                # Controls container
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.DatePickerRange(
                                        id="tab2-date-picker",
                                        display_format="DD/MM/YYYY",
                                        start_date=None,
                                        end_date=None,
                                        min_date_allowed=None,
                                        max_date_allowed=None,
                                        style={"marginRight": "10px"},
                                    ),
                                    dcc.Dropdown(
                                        id="tab2-company-selector",
                                        options=get_all_company_options(),
                                        value=[],
                                        multi=True,
                                        placeholder="Search and select companies... Click the refresh button to update the list.",
                                        style={"minWidth": "300px", "flex": "1"},
                                    ),
                                    dbc.Button(
                                        html.I(className="fas fa-sync-alt"),
                                        id="tab2-refresh-companies",
                                        color="light",
                                        size="sm",
                                        style={"marginLeft": "10px"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "10px",
                                    "width": "100%",
                                },
                            ),
                            width=12,
                        ),
                    ],
                    className="mb-2",
                    align="center",
                    justify="start",
                ),
                # Quick select buttons and row selector on the same row
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Last 7 days",
                                        id="tab2-quick-7",
                                        n_clicks=0,
                                        color="secondary",
                                    ),
                                    dbc.Button(
                                        "Last 30 days",
                                        id="tab2-quick-30",
                                        n_clicks=0,
                                        color="secondary",
                                    ),
                                    dbc.Button(
                                        "All time",
                                        id="tab2-quick-all",
                                        n_clicks=0,
                                        color="secondary",
                                    ),
                                ],
                                size="sm",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="tab2-page-size",
                                options=[
                                    {"label": str(n), "value": n}
                                    for n in [10, 20, 50, 100]
                                ],
                                value=20,
                                clearable=False,
                                style={"width": "100px"},
                            ),
                            width="auto",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "flex-end",
                            },
                        ),
                    ],
                    className="mb-3",
                    align="center",
                    justify="between",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                id="loading-tab2-table",
                                type="circle",
                                children=html.Div(id="tab2-table-container"),
                            ),
                            width=12,
                        ),
                    ]
                ),
            ],
            style={"padding": "30px"},
        ),
    ]
)


@app.callback(
    [
        ddep.Output("tab2-table-container", "children"),
        ddep.Output("tab2-date-picker", "min_date_allowed"),
        ddep.Output("tab2-date-picker", "max_date_allowed"),
        ddep.Output("tab2-date-picker", "start_date"),
        ddep.Output("tab2-date-picker", "end_date"),
    ],
    [
        ddep.Input("tab2-company-selector", "value"),
        ddep.Input("tab2-date-picker", "start_date"),
        ddep.Input("tab2-date-picker", "end_date"),
        ddep.Input("tab2-page-size", "value"),
        ddep.Input("tab2-quick-7", "n_clicks"),
        ddep.Input("tab2-quick-30", "n_clicks"),
        ddep.Input("tab2-quick-all", "n_clicks"),
    ],
)
def update_tab2_table(
    selected_companies,
    start_date_input,
    end_date_input,
    page_size,
    n7,
    n30,
    nall,
):
    ctx = callback_context
    triggered_id = (
        ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "initial load"
    )
    print(f"\n--- update_tab2_table triggered by: {triggered_id} ---")
    print(f"Selected Companies: {selected_companies}")
    print(f"Input Dates: start={start_date_input}, end={end_date_input}")

    min_date_allowed, max_date_allowed = get_company_date_range(selected_companies)

    if min_date_allowed is None or max_date_allowed is None:
        print("No valid date range found for selected companies.")
        min_date_allowed_out = datetime(1990, 1, 1).strftime("%Y-%m-%d")
        max_date_allowed_out = datetime.now().strftime("%Y-%m-%d")
        start_date_out = None
        end_date_out = None
        if not selected_companies:
            table_content = html.Div(
                "No company selected.",
                style={"textAlign": "center", "color": "#888", "padding": "30px"},
            )
        else:
            table_content = html.Div(
                "No data found for selected companies.",
                style={"textAlign": "center", "color": "#888", "padding": "30px"},
            )

        return (
            table_content,
            min_date_allowed_out,
            max_date_allowed_out,
            start_date_out,
            end_date_out,
        )

    start_date = start_date_input
    end_date = end_date_input

    if triggered_id == "tab2-company-selector":
        start_date = min_date_allowed
        end_date = max_date_allowed
    elif triggered_id == "tab2-quick-7":
        end_date = max_date_allowed
        start_date = end_date - pd.Timedelta(days=6)
    elif triggered_id == "tab2-quick-30":
        end_date = max_date_allowed
        start_date = end_date - pd.Timedelta(days=29)
    elif triggered_id == "tab2-quick-all":
        start_date = min_date_allowed
        end_date = max_date_allowed

    try:
        start_date = (
            pd.to_datetime(start_date, utc=True) if start_date else min_date_allowed
        )
        end_date = pd.to_datetime(end_date, utc=True) if end_date else max_date_allowed
    except Exception as e:
        print(f"Date parsing error: {e}. Falling back to allowed range.")
        start_date = min_date_allowed
        end_date = max_date_allowed

    start_date = max(start_date, min_date_allowed)
    end_date = min(end_date, max_date_allowed)
    if start_date > end_date:
        start_date = min_date_allowed

    print(f"Allowed Dates: min={min_date_allowed}, max={max_date_allowed}")
    print(f"Final Dates for Query: start={start_date}, end={end_date}")

    filtered_df = get_stock_data(selected_companies, start_date, end_date)

    if filtered_df.empty:
        print("Filtered DataFrame is empty.")
        table_content = html.Div(
            "No data available for the selected companies and date range.",
            style={"textAlign": "center", "color": "#888", "padding": "30px"},
        )
    else:
        print(f"Filtered DataFrame shape: {filtered_df.shape}")
        grouped = (
            filtered_df.groupby(["Date", "Cid", "Company"])
            .agg(
                Min=("Low", "min"),
                Max=("High", "max"),
                Open=("Open", "first"),
                Close=("Close", "last"),
                Mean=("Mean", "mean"),
                Std=("Close", "std"),
            )
            .reset_index()
            .sort_values(["Company", "Date"], ascending=[True, False])
        )
        grouped["Date"] = grouped["Date"].dt.strftime("%d/%m/%Y")
        for col in ["Min", "Max", "Open", "Close", "Mean", "Std"]:
            grouped[col] = grouped[col].round(2)
        grouped["Std"] = grouped["Std"].fillna(0)
        grouped["Mean"] = grouped["Mean"].fillna(0)

        table_content = dash_table.DataTable(
            columns=[
                {"name": "Date", "id": "Date"},
                {"name": "Company", "id": "Company"},
                {"name": "Min", "id": "Min"},
                {"name": "Max", "id": "Max"},
                {"name": "Open", "id": "Open"},
                {"name": "Close", "id": "Close"},
                {"name": "Mean", "id": "Mean"},
                {"name": "Std", "id": "Std"},
            ],
            data=grouped.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={
                "padding": "8px",
                "textAlign": "center",
                "fontFamily": "Poppins, Segoe UI, sans-serif",
            },
            style_header={
                "backgroundColor": "#f8f9fa",
                "fontWeight": "bold",
                "borderBottom": "2px solid #007bff",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#f9f9f9",
                }
            ],
            page_size=page_size,
            sort_action="native",
            sort_mode="multi",
        )

    start_date_out = start_date.strftime("%Y-%m-%d") if start_date else None
    end_date_out = end_date.strftime("%Y-%m-%d") if end_date else None
    min_date_allowed_out = (
        min_date_allowed.strftime("%Y-%m-%d") if min_date_allowed else None
    )
    max_date_allowed_out = (
        max_date_allowed.strftime("%Y-%m-%d") if max_date_allowed else None
    )

    print(
        f"Output Dates: start={start_date_out}, end={end_date_out}, min_allowed={min_date_allowed_out}, max_allowed={max_date_allowed_out}"
    )
    print("--- update_tab2_table finished ---")

    return (
        table_content,
        min_date_allowed_out,
        max_date_allowed_out,
        start_date_out,
        end_date_out,
    )

@app.callback(
    ddep.Output("tab2-company-selector", "options"),
    [ddep.Input("tab2-refresh-companies", "n_clicks")],
    prevent_initial_call=True
)
def refresh_companies_list(n_clicks):
    """
    Callback to refresh the companies list when the refresh button is clicked
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
        
    return get_all_company_options()
