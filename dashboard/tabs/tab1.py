import pandas as pd
from dash import dcc, html, ALL, callback_context, dash
import dash.dependencies as ddep
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import colorsys
from enum import Enum
import random

from app import app, db

theme = {
    "primary": "#007bff",  # Blue
    "secondary": "#6c757d",  # Gray
    "success": "#28a745",  # Green
    "danger": "#dc3545",  # Red
    "warning": "#ffc107",  # Yellow
    "info": "#17a2b8",  # Cyan
    "light": "#f8f9fa",  # Light gray
    "dark": "#343a40",  # Dark gray
    "background": "#f8f9fa",  # Light background
    "text": "#212529",  # Text color
}

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
    "Value",
    "CloseValue",
    "CompanyDisplay",
]


def get_stock_data(cids=None):
    """
    Fetches all historical stock data for a given list of company Cids.
    No date filtering is applied here.
    """
    print(f"--- get_stock_data called with cids: {cids} ---")

    if not cids:
        print("No CIDs provided, returning empty DataFrame.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if isinstance(cids, (int, str)):
        cids = [cids]
    print(f"Processed CIDs for query: {cids}")

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
            ds.mean as value,
            ds.close as close_value
        FROM companies c
        JOIN daystocks ds ON c.id = ds.cid
        WHERE c.id IN %(cids)s
        ORDER BY c.id, ds.date ASC
    """
    params = {"cids": tuple(cids)}

    print(f"Executing SQL Query:\n{query}")
    print(f"With Parameters: {params}")

    try:
        companies_df = db.df_query(query, params=params)
        print(
            f"Query returned DataFrame shape: {companies_df.shape}"
        )
    except Exception as e:
        print(f"!!! ERROR during db.df_query: {e}")
        companies_df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    if companies_df.empty:
        print("Query returned an empty DataFrame.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Rename columns
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
            "value": "Value",
            "close_value": "CloseValue",
        }
    )

    # Convert Date column
    companies_df["Date"] = pd.to_datetime(companies_df["Date"], utc=True)

    # Add CompanyDisplay column
    companies_df["CompanyDisplay"] = (
        companies_df["Company"] + " (" + companies_df["Symbol"] + ")"
    )

    # Ensure all expected columns are present
    for col in EXPECTED_COLUMNS:
        if col not in companies_df.columns:
            companies_df[col] = None

    print(
        f"--- get_stock_data finished, returning shape: {companies_df[EXPECTED_COLUMNS].shape} ---"
    )
    return companies_df[EXPECTED_COLUMNS]


def get_all_company_options():
    """
    Fetches all companies for populating the dropdown.
    """
    query = """
        SELECT 
            c.id as cid, 
            c.name as company, 
            c.symbol as symbol
        FROM companies c
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


class StabilityLevel(Enum):
    STABLE = {"label": "Stable", "color": "green"}
    SLIGHTLY_UNSTABLE = {"label": "Slightly Unstable", "color": "orange"}
    HIGHLY_UNSTABLE = {"label": "Highly Unstable", "color": "red"}


stable_threshold = 0.02  # < 2% daily stddev
slightly_unstable_threshold = 0.05  # < 5% daily stddev


def risk_level_from_vol(vol):
    if pd.isna(vol) or vol < stable_threshold:
        return StabilityLevel.STABLE
    elif vol < slightly_unstable_threshold:
        return StabilityLevel.SLIGHTLY_UNSTABLE
    else:
        return StabilityLevel.HIGHLY_UNSTABLE


# Generate random colors for companies
def generate_color_mapping(companies):
    random.seed(69)  # Set seed for reproducibility
    colors = []
    for _ in companies:  # low ranges for darker colors
        r = random.randint(1, 200)
        g = random.randint(1, 200)
        b = random.randint(1, 200)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return dict(zip(companies, colors))


# Function to generate complementary colors
def generate_complementary_color(hex_color, increase=True):
    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    # Adjust hue for complementary color
    if increase:
        h = (h + 0.33) % 1.0  # Shift hue for cold color (e.g., blue/green)
    else:
        h = (h - 0.33) % 1.0  # Shift hue for warm color (e.g., red/orange)
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# Update the calculate_bollinger_bands function to use rolling window
def calculate_bollinger_bands(df, num_std=2, window=20):
    # Ensure data is sorted chronologically
    df = df.sort_values("Date").copy()

    # Calculate rolling mean and standard deviation
    df["Moving Average"] = df["Close"].rolling(window=window, min_periods=1).mean()
    df["Std Dev"] = df["Close"].rolling(window=window, min_periods=1).std()

    # Calculate Upper and Lower Bands
    df["Upper Band"] = df["Moving Average"] + (df["Std Dev"] * num_std)
    df["Lower Band"] = df["Moving Average"] - (df["Std Dev"] * num_std)

    # Handle cases where Lower Band is negative
    df["Lower Band"] = df["Lower Band"].clip(lower=0)

    return df

# Footer for the layout
footer = html.Footer(
    html.P(
        [
            "© 2025 Ricou Bank • ",
            html.A("Contact", href="#"),
            " • ",
            html.A("Privacy", href="#"),
        ],
        style={"textAlign": "center", "color": theme["secondary"]},
    ),
    style={
        "padding": "20px",
        "marginTop": "40px",
        "borderTop": f"1px solid {theme['light']}",
    },
)

bollinger_state_store = dcc.Store(id="bollinger-state-store", data={})
xaxis_range_store = dcc.Store(id="xaxis-range-store", data=None)
date_range_store = dcc.Store(
    id="date-range-store",
    data=None,
)

# Wrapping main content in cards
tab1_layout = html.Div(
    [
        bollinger_state_store,
        xaxis_range_store,
        date_range_store,
        dbc.Card(
            dbc.CardBody(
                [
                    # Controls: Timeframe, Chart Type, and Bollinger Window
                    html.Div(
                        [
                            # Timeframe Selector
                            html.Div(
                                [
                                    html.H5(
                                        "Select Timeframe",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.DatePickerRange(
                                        id="date-picker",
                                        display_format="DD/MM/YYYY",
                                        style={"width": "300px"},
                                    ),
                                ],
                                style={"marginRight": "20px"},
                            ),
                            # Chart Type Selection as Toggle Buttons
                            html.Div(
                                [
                                    html.H5(
                                        "Chart Type", style={"marginBottom": "10px"}
                                    ),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Line",
                                                id="line-button",
                                                n_clicks=0,
                                                color="primary",
                                                outline=True,
                                                className="mr-2",
                                                size="md",
                                            ),
                                            dbc.Tooltip(
                                                "Toggle the chart type between line and candlestick",
                                                target="line-button",
                                                placement="top",
                                            ),
                                            dbc.Button(
                                                "Candlestick",
                                                id="candlestick-button",
                                                n_clicks=0,
                                                style={"width": "150px"},
                                            ),
                                            dbc.Tooltip(
                                                "Display stock data as candlesticks",
                                                target="candlestick-button",
                                                placement="top",
                                            ),
                                        ],
                                        size="md",
                                    ),
                                ],
                                style={"marginRight": "20px"},
                            ),
                            # Bollinger Window Selector
                            html.Div(
                                [
                                    html.H5(
                                        "Bollinger Window Size",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Slider(
                                        id="bollinger-window-slider",
                                        min=3,
                                        max=50,
                                        step=1,
                                        value=20,  # Default window size
                                        marks={i: str(i) for i in range(3, 51, 5)},
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                        className="bollinger-slider",
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "marginBottom": "20px",
                        },
                    ),
                ]
            ),
            style={"marginBottom": "20px"},
            className="shadow-sm",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="loading-graph",
                                        type="circle",
                                        children=[
                                            dcc.Graph(
                                                id="stock-graph",
                                                config={"displayModeBar": False},
                                            )
                                        ],
                                    )
                                ]
                            ),
                            className="shadow-sm",
                        )
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Company Selector", className="card-title"),
                                    dcc.Dropdown(
                                        id="company-selector",
                                        options=get_all_company_options(),
                                        value=[],
                                        multi=True,
                                        placeholder="Search and select companies...",
                                        style={"marginBottom": "20px"},
                                        className="hide-dropdown-selection",
                                    ),
                                    html.Div(
                                        id="company-table",
                                        style={
                                            "border": "1px solid #ccc",
                                            "padding": "10px",
                                            "borderRadius": "5px",
                                        },
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        )
                    ],
                    width=4,
                ),
            ]
        ),
        footer,
    ],
    style={"backgroundColor": theme["background"], "padding": "10px"},
)


# Callback to handle chart type toggle and update the graph
@app.callback(
    [
        ddep.Output("line-button", "style"),
        ddep.Output("candlestick-button", "style"),
        ddep.Output("stock-graph", "figure"),
        ddep.Output("date-picker", "start_date"),
        ddep.Output("date-picker", "end_date"),
    ],
    [
        ddep.Input("line-button", "n_clicks"),
        ddep.Input("candlestick-button", "n_clicks"),
        ddep.Input("company-selector", "value"),
        ddep.Input("date-picker", "start_date"),
        ddep.Input("date-picker", "end_date"),
        ddep.Input("stock-graph", "relayoutData"),
        ddep.Input({"type": "bollinger-checkbox", "index": ALL}, "value"),
        ddep.Input("bollinger-state-store", "data"),
        ddep.Input("bollinger-window-slider", "value"),
    ],
    [
        ddep.State({"type": "bollinger-checkbox", "index": ALL}, "id"),
    ],
)
def update_chart(
    line_clicks,
    candlestick_clicks,
    selected_companies,
    start_date_input,
    end_date_input,
    relayout_data,
    bollinger_values,
    bollinger_state,
    window_size,
    checkbox_ids,
):
    print("\n--- update_chart triggered ---")
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else "Initial load"
    print(f"Triggered by: {triggered_id}")

    start_date = start_date_input
    end_date = end_date_input

    if relayout_data:
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            start_date = pd.to_datetime(relayout_data["xaxis.range[0]"], utc=True)
            end_date = pd.to_datetime(relayout_data["xaxis.range[1]"], utc=True)
            start_date_input = start_date.strftime("%Y-%m-%d")
            end_date_input = end_date.strftime("%Y-%m-%d")
        elif "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
            start_date = None
            end_date = None
            start_date_input = None
            end_date_input = None
    else:
        start_date = (
            pd.to_datetime(start_date_input, utc=True) if start_date_input else None
        )
        end_date = pd.to_datetime(end_date_input, utc=True) if end_date_input else None

    print(f"Initial Dates: start={start_date_input}, end={end_date_input}")
    print(f"Dates after relayout/parse: start={start_date}, end={end_date}")

    if selected_companies is None:
        selected_companies = []
    elif isinstance(selected_companies, (int, str)):
        selected_companies = [selected_companies]
    print(f"Selected Company CIDs: {selected_companies}")

    if line_clicks is None:
        line_clicks = 0
    if candlestick_clicks is None:
        candlestick_clicks = 0
    if line_clicks >= candlestick_clicks:
        chart_type = "line"
        line_style = {
            "backgroundColor": theme["success"],
            "color": "white",
            "width": "100px",
        }
        candlestick_style = {
            "backgroundColor": "white",
            "color": "black",
            "width": "150px",
        }
        graph_title = "Stock Price (Line)"
    else:
        chart_type = "candlestick"
        line_style = {"backgroundColor": "white", "color": "black", "width": "100px"}
        candlestick_style = {
            "backgroundColor": theme["success"],
            "color": "white",
            "width": "150px",
        }
        graph_title = "Stock Price (Candlestick)"

    print(f"Chart Type: {chart_type}")

    print("Fetching data...")
    fetched_df = get_stock_data(selected_companies)
    print(f"Fetched DataFrame shape: {fetched_df.shape}")
    if not fetched_df.empty:
        print("Fetched DataFrame head:\n", fetched_df.head())
    else:
        print("Fetched DataFrame is EMPTY.")
        # Early exit if no data fetched at all
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected companies",
            xaxis_title="Time",
            yaxis_title="Stock Value",
        )
        return line_style, candlestick_style, fig, start_date_input, end_date_input

    print("Filtering data by date...")
    if start_date and end_date:
        start_date_dt = pd.to_datetime(start_date, utc=True)
        end_date_dt = pd.to_datetime(end_date, utc=True)
        print(f"Filtering between {start_date_dt} and {end_date_dt}")
        filtered_df = fetched_df[
            (fetched_df["Date"] >= start_date_dt) & (fetched_df["Date"] <= end_date_dt)
        ].copy()
    else:
        print("No date range specified, using all fetched data.")
        filtered_df = fetched_df.copy()
        if not filtered_df.empty:
            start_date_input = filtered_df["Date"].min().strftime("%Y-%m-%d")
            end_date_input = filtered_df["Date"].max().strftime("%Y-%m-%d")

    print(f"Filtered DataFrame shape: {filtered_df.shape}")
    if not filtered_df.empty:
        print("Filtered DataFrame head:\n", filtered_df.head())
    else:
        print("Filtered DataFrame is EMPTY after date filtering.")
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected date range",
            xaxis_title="Time",
            yaxis_title="Stock Value",
        )
        return line_style, candlestick_style, fig, start_date_input, end_date_input

    print("Calculating Return and Volatility...")
    filtered_df["Return"] = filtered_df.groupby("Cid")["Close"].pct_change()
    volatility = filtered_df.groupby("Cid")["Return"].std()
    print(f"Volatility calculated:\n{volatility}")

    color_mapping = generate_color_mapping(selected_companies)
    print(f"Color Mapping: {color_mapping}")

    companies_with_bollinger = []
    bollinger_state = bollinger_state or {}
    active_checkbox_cids = set()
    if checkbox_ids and bollinger_values:
        for i, checkbox_id in enumerate(checkbox_ids):
            cid_str = checkbox_id["index"]
            if i < len(bollinger_values) and bollinger_values[i]:
                try:
                    active_checkbox_cids.add(int(cid_str))
                except ValueError:
                    pass

    for cid in selected_companies:
        if str(cid) in bollinger_state and bollinger_state[str(cid)]:
            companies_with_bollinger.append(cid)
        elif cid in active_checkbox_cids and cid not in companies_with_bollinger:
            companies_with_bollinger.append(cid)

    print(f"Companies with Bollinger Bands: {companies_with_bollinger}")
    print(f"Bollinger Window Size: {window_size}")

    print("Generating chart traces...")
    fig = go.Figure()

    for cid in selected_companies:
        print(f"Processing CID: {cid}")
        company_data_filtered = filtered_df[filtered_df["Cid"] == cid]
        print(
            f"  Data shape for CID {cid} after filtering: {company_data_filtered.shape}"
        )

        if company_data_filtered.empty:
            print(f"  Skipping CID {cid} - no data in filtered range.")
            continue

        company_display_row = fetched_df[fetched_df["Cid"] == cid]
        if not company_display_row.empty:
            company_display = company_display_row["CompanyDisplay"].iloc[0]
        else:
            company_display = f"Company {cid}"

        company_color = color_mapping.get(cid, "#000000")
        print(f"  Display Name: {company_display}, Color: {company_color}")

        if chart_type == "line":
            print(f"  Adding Line trace for CID {cid}")
            fig.add_trace(
                go.Scatter(
                    x=company_data_filtered["Date"],
                    y=company_data_filtered["Close"],
                    mode="lines",
                    name=company_display,
                    line=dict(color=company_color),
                )
            )
        elif chart_type == "candlestick":
            print(f"  Adding Candlestick trace for CID {cid}")
            fig.add_trace(
                go.Candlestick(
                    x=company_data_filtered["Date"],
                    open=company_data_filtered["Open"],
                    high=company_data_filtered["High"],
                    low=company_data_filtered["Low"],
                    close=company_data_filtered["Close"],
                    name=company_display,
                    increasing_line_color=company_color,
                    decreasing_line_color=generate_complementary_color(
                        company_color, increase=False
                    ),
                )
            )

        if cid in companies_with_bollinger:
            print(f"  Calculating and adding Bollinger Bands for CID {cid}")
            company_data_bollinger = calculate_bollinger_bands(
                company_data_filtered, num_std=2, window=window_size
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data_bollinger["Date"],
                    y=company_data_bollinger["Upper Band"],
                    mode="lines",
                    name=f"{company_display} Upper",
                    line=dict(color=company_color, width=1, dash="dot"),
                    opacity=0.7,
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data_bollinger["Date"],
                    y=company_data_bollinger["Lower Band"],
                    mode="lines",
                    name=f"{company_display} Lower",
                    line=dict(color=company_color, width=1, dash="dot"),
                    opacity=0.3,
                    showlegend=True,
                    fill="tonexty",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data_bollinger["Date"],
                    y=company_data_bollinger["Moving Average"],
                    mode="lines",
                    name=f"{company_display} MA",
                    line=dict(color=company_color, width=1.5, dash="dash"),
                    opacity=0.6,
                    showlegend=True,
                )
            )
        else:
            print(f"  Bollinger Bands not enabled for CID {cid}")

    print(f"Total traces added to figure: {len(fig.data)}")

    print("Updating figure layout...")
    fig.update_layout(
        title=graph_title,
        xaxis_title="Time",
        yaxis_title="Stock Value (Log Scale)",
        template="plotly_white",
        title_x=0.5,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis=dict(
            rangeslider=dict(
                visible=False,
                thickness=0.05,
                bgcolor="#F5F5F5",
            ),
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ]
                ),
                bgcolor="#E9ECEF",
                activecolor=theme["primary"],
                y=1.1,
            ),
            type="date",
            calendar="gregorian",
        ),
        yaxis_type="log",
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            namelength=-1,
        ),
        hovermode="x unified",
    )

    print("--- update_chart finished ---\n")
    return line_style, candlestick_style, fig, start_date_input, end_date_input


@app.callback(
    [
        ddep.Output("company-table", "children"),
        ddep.Output("company-selector", "options"),
        ddep.Output("company-selector", "value"),
    ],
    [
        ddep.Input("company-selector", "value"),
        ddep.Input({"type": "delete-button", "index": ALL}, "n_clicks"),
        ddep.Input("date-picker", "start_date"),
        ddep.Input("date-picker", "end_date"),
    ],
    [
        ddep.State("bollinger-state-store", "data"),
    ],
)
def update_company_table(
    selected_companies, delete_clicks, start_date, end_date, bollinger_state
):
    if selected_companies is None:
        selected_companies = []
    elif isinstance(selected_companies, (int, str)):
        selected_companies = [selected_companies]

    bollinger_state = bollinger_state or {}

    ctx = callback_context
    triggered_dict = {}
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"]
        if "{" in prop_id and "}" in prop_id:
            try:
                import json

                json_part = prop_id[prop_id.find("{") : prop_id.rfind("}") + 1]
                triggered_dict = json.loads(json_part)
            except json.JSONDecodeError as e:
                triggered_dict = {}
        else:
            triggered_id = prop_id.split(".")[0]

        if triggered_dict.get("type") == "delete-button":
            cid_to_remove_str = triggered_dict.get("index")
            if cid_to_remove_str:
                try:
                    cid_to_remove = int(cid_to_remove_str)
                    if cid_to_remove in selected_companies:
                        selected_companies.remove(cid_to_remove)
                except ValueError:
                    pass

    fetched_df = get_stock_data(selected_companies)

    if start_date and end_date:
        start_date_dt = pd.to_datetime(start_date, utc=True)
        end_date_dt = pd.to_datetime(end_date, utc=True)
        filtered_df_for_volatility = fetched_df[
            (fetched_df["Date"] >= start_date_dt) & (fetched_df["Date"] <= end_date_dt)
        ].copy()
    else:
        filtered_df_for_volatility = fetched_df.copy()

    if not filtered_df_for_volatility.empty:
        filtered_df_for_volatility["Return"] = filtered_df_for_volatility.groupby(
            "Cid"
        )["Close"].pct_change()
        volatility = filtered_df_for_volatility.groupby("Cid")["Return"].std()
    else:
        volatility = pd.Series(dtype=float)

    stability_mapping = {
        cid: risk_level_from_vol(volatility.get(cid, None))
        for cid in selected_companies
    }

    all_options = get_all_company_options()

    if selected_companies:
        table_header = html.Div(
            children=[
                html.Div(
                    "Stability",
                    style={
                        "width": "20%",
                        "fontWeight": "bold",
                        "textAlign": "left",
                        "padding": "8px",
                        "whiteSpace": "nowrap",
                    },
                ),
                html.Div(
                    "Company",
                    style={
                        "width": "35%",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "padding": "8px",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                    },
                ),
                html.Div(
                    "Bollinger",
                    style={
                        "width": "25%",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "padding": "8px",
                        "whiteSpace": "nowrap",
                    },
                ),
                html.Div(
                    "Delete",
                    style={
                        "width": "20%",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "padding": "8px",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "borderBottom": f"3px solid {theme['primary']}",
                "backgroundColor": theme["light"],
                "borderRadius": "8px 8px 0 0",
                "padding": "10px 5px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
        )

        table_rows = []
        for i, cid in enumerate(selected_companies):
            row_bg_color = "#f9f9f9" if i % 2 == 0 else "white"
            stability = stability_mapping.get(cid, StabilityLevel.STABLE)
            is_bollinger_enabled = bollinger_state.get(str(cid), False)

            company_info = fetched_df[fetched_df["Cid"] == cid]
            if not company_info.empty:
                name = company_info.iloc[0]["CompanyDisplay"]
            else:
                name = f"Company {cid}"

            table_rows.append(
                html.Div(
                    [
                        html.Div(
                            html.Span(
                                stability.value["label"],
                                style={
                                    "backgroundColor": stability.value["color"],
                                    "color": "white",
                                    "padding": "3px 8px",
                                    "borderRadius": "12px",
                                    "fontSize": "0.85rem",
                                    "display": "inline-block",
                                },
                            ),
                            style={
                                "width": "20%",
                                "textAlign": "left",
                                "padding": "8px",
                            },
                        ),
                        html.Div(
                            name,
                            style={
                                "width": "35%",
                                "textAlign": "center",
                                "padding": "8px",
                                "fontWeight": "500",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(
                            dcc.Checklist(
                                options=[{"label": "", "value": "bollinger"}],
                                id={"type": "bollinger-checkbox", "index": str(cid)},
                                value=(["bollinger"] if is_bollinger_enabled else []),
                                className="bollinger-checkbox",
                                style={"marginLeft": "10px", "transform": "scale(1.2)"},
                            ),
                            style={
                                "width": "25%",
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                                "padding": "8px",
                            },
                        ),
                        html.Div(
                            dbc.Button(
                                "✖",
                                id={"type": "delete-button", "index": str(cid)},
                                color="danger",
                                size="sm",
                                style={
                                    "width": "30px",
                                    "height": "30px",
                                    "padding": "2px 0",
                                    "borderRadius": "50%",
                                    "fontWeight": "bold",
                                    "boxShadow": "0 2px 5px rgba(220,53,69,0.3)",
                                    "transition": "all 0.2s ease",
                                },
                            ),
                            style={
                                "width": "20%",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "padding": "5px",
                        "borderBottom": "1px solid #eee",
                        "backgroundColor": row_bg_color,
                        "transition": "background-color 0.2s",
                        "borderRadius": (
                            "0 0 8px 8px" if i == len(selected_companies) - 1 else "0"
                        ),
                    },
                )
            )

        table_content = [table_header] + table_rows
    else:
        table_content = html.Div(
            [
                html.Span("ℹ️", style={"marginRight": "5px"}),
                "No companies selected. Select companies from the dropdown above.",
            ],
            style={
                "textAlign": "center",
                "padding": "20px 10px",
                "color": theme["secondary"],
                "backgroundColor": theme["light"],
                "borderRadius": "8px",
                "border": f"1px dashed {theme['secondary']}",
            },
        )

    return table_content, all_options, selected_companies


@app.callback(
    ddep.Output("bollinger-state-store", "data"),
    [
        ddep.Input({"type": "bollinger-checkbox", "index": ALL}, "value"),
        ddep.Input({"type": "bollinger-checkbox", "index": ALL}, "id"),
    ],
    [ddep.State("bollinger-state-store", "data")],
)
def update_bollinger_state(checkbox_values, checkbox_ids, current_state):
    if not current_state:
        current_state = {}

    for i, checkbox_id in enumerate(checkbox_ids):
        cid_str = checkbox_id["index"]
        value = checkbox_values[i] if i < len(checkbox_values) else []
        current_state[cid_str] = bool(value)

    return current_state


@app.callback(
    ddep.Output("xaxis-range-store", "data"),
    [
        ddep.Input("date-picker", "start_date"),
        ddep.Input("date-picker", "end_date"),
        ddep.Input("stock-graph", "relayoutData"),
    ],
    [ddep.State("xaxis-range-store", "data")],
    prevent_initial_call=True,
)
def sync_date_inputs(start_date_picker, end_date_picker, relayout_data, current_range):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "stock-graph" and relayout_data:
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            return [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
        elif "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
            return None
    elif triggered_id == "date-picker":
        return [start_date_picker, end_date_picker]

    return current_range
