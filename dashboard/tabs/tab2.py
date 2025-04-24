import pandas as pd

from dash import dcc, html, dash_table, callback_context
import dash.dependencies as ddep
import dash_bootstrap_components as dbc
import plotly.express as px

from app import app, db


def get_stock_data():
    companies = db.df_query(
        """
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
        ORDER BY ds.date DESC
        """
    )
    companies["date"] = pd.to_datetime(companies["date"], utc=True)
    companies = companies.rename(
        columns={
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
    return companies


df = get_stock_data()

tab2_layout = html.Div(
    [
        html.Div(
            [
                html.H3("Raw data", style={"marginBottom": "20px"}),
                # Controls container
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.DatePickerRange(
                                        id="tab2-date-picker",
                                        min_date_allowed=df["Date"].min(),
                                        max_date_allowed=df["Date"].max(),
                                        start_date=df["Date"].min(),
                                        end_date=df["Date"].max(),
                                        display_format="DD/MM/YYYY",
                                        style={"marginRight": "10px"},
                                    ),
                                    dcc.Dropdown(
                                        id="tab2-company-selector",
                                        options=[
                                            {
                                                "label": f"{row['Company']} ({row['Symbol']})",
                                                "value": row["Company"],
                                            }
                                            for _, row in df.drop_duplicates("Company")
                                            .sort_values("Company")
                                            .iterrows()
                                        ],
                                        value=[],
                                        multi=True,
                                        placeholder="Search and select companies...",
                                        style={"minWidth": "300px", "flex": "1"},
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
                            html.Div(id="tab2-table-container"),
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
    [
        ddep.State("tab2-date-picker", "min_date_allowed"),
        ddep.State("tab2-date-picker", "max_date_allowed"),
    ],
)
def update_tab2_table(
    selected_companies,
    start_date,
    end_date,
    page_size,
    n7,
    n30,
    nall,
    min_date,
    max_date,
):
    ctx = callback_context
    # Quick select logic
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "tab2-quick-7":
            end_date = pd.to_datetime(max_date)
            start_date = end_date - pd.Timedelta(days=6)
        elif trigger == "tab2-quick-30":
            end_date = pd.to_datetime(max_date)
            start_date = end_date - pd.Timedelta(days=29)
        elif trigger == "tab2-quick-all":
            start_date = df["Date"].min()
            end_date = df["Date"].max()

    # Ensure start_date and end_date are timezone-aware (UTC)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("UTC")

    if not selected_companies:
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        return (
            html.Div(
                "No company selected.",
                style={"textAlign": "center", "color": "#888", "padding": "30px"},
            ),
            min_date,
            max_date,
            min_date,
            max_date,
        )

    filtered = df[
        (df["Company"].isin(selected_companies))
        & (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ].copy()

    if filtered.empty:
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        return (
            html.Div(
                "No data for the selection.",
                style={"textAlign": "center", "color": "#888", "padding": "30px"},
            ),
            min_date,
            max_date,
            min_date,
            max_date,
        )

    # Group by Date and Company, aggregate
    grouped = (
        filtered.groupby(["Date", "Company"])
        .agg(
            Min=("Low", "min"),
            Max=("High", "max"),
            Open=("Open", "first"),
            Close=("Close", "last"),
            Mean=("Mean", "mean"),
            Std=("Close", "std"),
        )
        .reset_index()
        .sort_values(["Company", "Date"])
    )
    grouped["Date"] = grouped["Date"].dt.strftime("%d/%m/%Y")
    for col in ["Min", "Max", "Open", "Close", "Mean", "Std"]:
        grouped[col] = grouped[col].round(2)
    grouped["Std"] = grouped["Std"].fillna(0)
    grouped["Mean"] = grouped["Mean"].fillna(0)

    min_date = filtered["Date"].min()
    max_date = filtered["Date"].max()

    return (
        dash_table.DataTable(
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
        ),
        min_date,
        max_date,
        start_date,
        end_date,
    )
