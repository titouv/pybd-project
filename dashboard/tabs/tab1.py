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


def get_stock_data():
    companies = db.df_query(
        """
        SELECT 
            c.id as cid, 
            c.name as company, 
            c.symbol as symbol,  -- Add this line
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
            "value": "Value",
            "close_value": "CloseValue",
        }
    )
    return companies


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


# Get initial data
df = get_stock_data()
df["CompanyDisplay"] = df["Company"] + " (" + df["Symbol"] + ")"

# Calculate daily returns for each company (needed for callbacks)
df = df.sort_values(["Company", "Date"])
df["Return"] = df.groupby("Company")["Close"].pct_change()


# Generate random colors for companies
def generate_color_mapping(companies):
    random.seed(69)  # Set seed for reproducibility
    colors = []
    for _ in companies:  # low ranges for darker colors
        r = random.randint(10, 100)
        g = random.randint(100, 100)
        b = random.randint(10, 100)
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


# Dynamically generate color mapping
color_mapping = generate_color_mapping(df["Company"].unique())

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
    data={"start_date": df["Date"].min(), "end_date": df["Date"].max()},
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
                                        min_date_allowed=df["Date"].min(),
                                        max_date_allowed=df["Date"].max(),
                                        start_date=df["Date"].min(),
                                        end_date=df["Date"].max(),
                                        display_format="YYYY-MM-DD",
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
                                        options=[
                                            {
                                                "label": row["CompanyDisplay"],
                                                "value": row["Company"],
                                            }
                                            for _, row in df.drop_duplicates(
                                                "Company"
                                            ).iterrows()
                                        ],
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
    style={"backgroundColor": theme["background"], "padding": "0px"},
)


# Callback to handle chart type toggle and update the graph with Bollinger Bands
@app.callback(
    [
        ddep.Output("line-button", "style"),
        ddep.Output("candlestick-button", "style"),
        ddep.Output("stock-graph", "figure"),
    ],
    [
        ddep.Input("line-button", "n_clicks"),
        ddep.Input("candlestick-button", "n_clicks"),
        ddep.Input("company-selector", "value"),
        ddep.Input("date-picker", "start_date"),
        ddep.Input("date-picker", "end_date"),
        ddep.Input({"type": "bollinger-checkbox", "index": ALL}, "value"),
        ddep.Input("bollinger-state-store", "data"),
        ddep.Input("bollinger-window-slider", "value"),
    ],
    [
        ddep.State({"type": "bollinger-checkbox", "index": ALL}, "id"),
        ddep.State("xaxis-range-store", "data"),
    ],
)
def update_chart(
    line_clicks,
    candlestick_clicks,
    selected_companies,
    start_date,
    end_date,
    bollinger_values,
    bollinger_state,
    window_size,
    checkbox_ids,
    xaxis_range,
):
    print("Callback triggered!")
    print("Selected companies:", selected_companies)
    print("Start date:", start_date)
    print("End date:", end_date)

    # Get current figure if it exists
    ctx = callback_context

    # Determine which button was clicked
    if line_clicks >= candlestick_clicks:
        chart_type = "line"
        line_style = {"backgroundColor": "green", "color": "white", "width": "100px"}
        candlestick_style = {
            "backgroundColor": "white",
            "color": "black",
            "width": "150px",
        }
        graph_title = "Linear Stock Chart"
    else:
        chart_type = "candlestick"
        line_style = {"backgroundColor": "white", "color": "black", "width": "100px"}
        candlestick_style = {
            "backgroundColor": "green",
            "color": "white",
            "width": "150px",
        }
        graph_title = "Candlestick Stock Chart"

    # Use stored range if available and not changing date picker
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if xaxis_range and not triggered.startswith("date-picker"):
        display_range = xaxis_range
    else:
        display_range = [start_date, end_date]

    # Preserve the current view's time range
    if start_date is None or end_date is None:
        start_date = df["Date"].min()
        end_date = df["Date"].max()
    else:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if not pd.Timestamp(start_date).tzinfo:
            start_date = pd.to_datetime(start_date).tz_localize("UTC")
        if not pd.Timestamp(end_date).tzinfo:
            end_date = pd.to_datetime(end_date).tz_localize("UTC")

    # Filter data based on selected companies and timeframe
    filtered_df = (
        df[
            (df["Company"].isin(selected_companies))
            & (df["Date"] >= start_date)
            & (df["Date"] <= end_date)
        ]
        .sort_values(["Company", "Date"])
        .copy()
    )

    filtered_df["Return"] = filtered_df.groupby("Company")["Close"].pct_change()
    volatility = filtered_df.groupby("Company")["Return"].std()

    if len(filtered_df) > 0:
        print("Data range:", filtered_df["Date"].min(), "to", filtered_df["Date"].max())
        print(
            "Sample data for first company:",
            filtered_df[filtered_df["Company"] == selected_companies[0]].head(),
        )
    else:
        print("No data in filtered_df")

    # Print volatility for each selected company (log on every chart update)
    for company in selected_companies:
        vol = volatility.get(company, None)
        print(f"[Chart] Volatility for {company}: {vol}")

    # Combine checkbox values and stored state
    companies_with_bollinger = []

    # Add companies from checkbox values
    if checkbox_ids and bollinger_values:
        for i, checkbox_id in enumerate(checkbox_ids):
            company = checkbox_id["index"]
            if (
                i < len(bollinger_values)
                and bollinger_values[i]
                and company in selected_companies
            ):
                companies_with_bollinger.append(company)

    # Add companies from stored state for any that might be missing from checkbox values
    # (this handles situations where the checkbox might not be rendered yet)
    for company in selected_companies:
        if (
            bollinger_state
            and company in bollinger_state
            and company not in companies_with_bollinger
        ):
            companies_with_bollinger.append(company)

    # Generate the appropriate chart
    fig = go.Figure()

    for company in selected_companies:
        company_data = filtered_df[filtered_df["Company"] == company]
        company_display = df[df["Company"] == company]["CompanyDisplay"].iloc[0]

        # Main price trace
        if chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Close"],
                    mode="lines",
                    name=company_display,  # Only the display name
                    line=dict(color=color_mapping[company]),
                )
            )
        elif chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=company_data["Date"],
                    open=company_data["Open"],
                    high=company_data["High"],
                    low=company_data["Low"],
                    close=company_data["Close"],
                    name=company_display,
                    increasing_line_color=color_mapping[company],
                    decreasing_line_color=generate_complementary_color(
                        color_mapping[company], increase=False
                    ),
                )
            )

        # Bollinger Bands
        if company in companies_with_bollinger:
            company_data = calculate_bollinger_bands(
                company_data, num_std=2, window=window_size
            )  # Use the selected window size for Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Upper Band"],
                    mode="lines",
                    name=f"{company_display} Upper Band",
                    line=dict(color=color_mapping[company], width=1, dash="dot"),
                    opacity=0.7,
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Moving Average"],
                    mode="lines",
                    name=f"{company_display} MA",
                    line=dict(color=color_mapping[company], width=1, dash="dash"),
                    opacity=0.5,
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Lower Band"],
                    mode="lines",
                    name=f"{company_display} Lower Band",
                    line=dict(color=color_mapping[company], width=1, dash="dot"),
                    opacity=0.3,
                    showlegend=True,
                    fill="tonexty",
                )
            )

    fig.update_layout(
        title=graph_title,
        xaxis_title="Time",
        yaxis_title="Stock Value (In USD)",
        template="plotly_white",
        title_x=0.5,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor="#F5F5F5",
            ),
            range=display_range,  # Use the preserved range
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
            type="date",  # Ensure proper date handling
            calendar="gregorian",
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            namelength=-1,  # Never truncate the name in hover
        ),
        hovermode="x unified",
    )

    return line_style, candlestick_style, fig


# Callback to update the company table and dropdown options
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
    # Initialize selected_companies as an empty list if None
    if selected_companies is None:
        selected_companies = []

    # Initialize bollinger_state if None
    if bollinger_state is None:
        bollinger_state = {}

    ctx = callback_context

    if ctx.triggered:
        print("PropID:", ctx.triggered[0]["prop_id"])

        # Split the prop_id and rejoin all parts except the last one
        prop_id_parts = ctx.triggered[0]["prop_id"].split(".")
        triggered_id = ".".join(prop_id_parts[:-1])  # Component ID
        triggered_property = prop_id_parts[-1]  # Property

        print("Triggered ID (raw):", triggered_id)
        print("Triggered Property:", triggered_property)

        # Safely parse the triggered_id
        try:
            import json

            triggered_dict = json.loads(triggered_id)
        except json.JSONDecodeError as e:
            print("Error parsing triggered_id:", e)
            triggered_dict = {}

        print("Triggered Dict:", triggered_dict)

        # Handle delete button click
        if (
            isinstance(triggered_dict, dict)
            and triggered_dict.get("type") == "delete-button"
        ):
            company_to_remove = triggered_dict.get("index")
            print("Company to Remove:", company_to_remove)
            print("Selected Companies Before Removal:", selected_companies)

            # Normalize company names for comparison
            selected_companies = [
                company.strip() for company in selected_companies
            ]  # Remove extra spaces
            if company_to_remove:
                company_to_remove = company_to_remove.strip()

                if company_to_remove in selected_companies:
                    selected_companies.remove(company_to_remove)
                    print("Selected Companies After Removal:", selected_companies)

    # Filter df for the current timeframe
    filtered_df = (
        df[
            (df["Company"].isin(selected_companies))
            & (df["Date"] >= start_date)
            & (df["Date"] <= end_date)
        ]
        .sort_values(["Company", "Date"])
        .copy()
    )

    filtered_df["Return"] = filtered_df.groupby("Company")["Close"].pct_change()
    volatility = filtered_df.groupby("Company")["Return"].std()

    # Print volatility for each selected company
    for company in selected_companies:
        vol = volatility.get(company, None)
        print(f"Volatility for {company}: {vol}")

    stability_mapping = {
        company: risk_level_from_vol(volatility.get(company, None))
        for company in selected_companies
    }

    print("Updated stability mapping:", stability_mapping)

    # Update dropdown options to exclude selected companies
    all_options = [
        {
            "label": row["CompanyDisplay"],
            "value": row["Company"],
        }
        for _, row in df.drop_duplicates("Company").iterrows()
    ]

    # Generate the company table
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
        for i, company in enumerate(selected_companies):
            # Alternate row background colors for better readability
            row_bg_color = "#f9f9f9" if i % 2 == 0 else "white"

            # Get stability info
            stability = stability_mapping.get(company, StabilityLevel.STABLE)

            # Check if this company has Bollinger Bands enabled in the stored state
            is_bollinger_enabled = bollinger_state.get(company, False)

            name = df[df["Company"] == company]["CompanyDisplay"].iloc[0]

            table_rows.append(
                html.Div(
                    [
                        # Stability
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
                        # Company Name
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
                        # Bollinger Checkbox - use the stored state
                        html.Div(
                            dcc.Checklist(
                                options=[{"label": "", "value": "bollinger"}],
                                id={"type": "bollinger-checkbox", "index": company},
                                value=(
                                    ["bollinger"] if is_bollinger_enabled else []
                                ),  # Set from stored state
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
                        # Delete Button
                        html.Div(
                            dbc.Button(
                                "✖",
                                id={"type": "delete-button", "index": company},
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
        # Display a placeholder message if no companies are selected
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

    # Update company table container styling
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
    # Initialize state if it doesn't exist
    if not current_state:
        current_state = {}

    # Update state with new checkbox values
    for i, checkbox_id in enumerate(checkbox_ids):
        company = checkbox_id["index"]
        value = checkbox_values[i]
        if value:
            current_state[company] = True
        else:
            # Only remove from state if it exists
            current_state.pop(company, None)

    return current_state


@app.callback(
    ddep.Output("xaxis-range-store", "data"),
    [
        ddep.Input("date-picker", "start_date"),
        ddep.Input("date-picker", "end_date"),
        ddep.Input("stock-graph", "relayoutData"),
    ],
    [ddep.State("xaxis-range-store", "data")],
)
def update_xaxis_range(start_date, end_date, relayout_data, current_range):
    # If the slider or range selector is used
    if (
        relayout_data
        and "xaxis.range[0]" in relayout_data
        and "xaxis.range[1]" in relayout_data
    ):
        return [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
    # If the date picker is used
    return [start_date, end_date]


@app.callback(
    [
        ddep.Output("date-picker", "start_date"),
        ddep.Output("date-picker", "end_date"),
        ddep.Output("date-range-store", "data"),
    ],
    [
        ddep.Input("stock-graph", "relayoutData"),
        ddep.Input("date-picker", "start_date"),
        ddep.Input("date-picker", "end_date"),
    ],
    [ddep.State("date-range-store", "data")],
)
def sync_date_range(relayout_data, picker_start_date, picker_end_date, current_range):
    # If the slider or range selector is used
    if (
        relayout_data
        and "xaxis.range[0]" in relayout_data
        and "xaxis.range[1]" in relayout_data
    ):
        start_date = pd.to_datetime(relayout_data["xaxis.range[0]"])
        end_date = pd.to_datetime(relayout_data["xaxis.range[1]"])
    else:
        # If the date picker is used
        start_date = pd.to_datetime(picker_start_date)
        end_date = pd.to_datetime(picker_end_date)

    # Update the store with the new range
    return start_date, end_date, {"start_date": start_date, "end_date": end_date}
