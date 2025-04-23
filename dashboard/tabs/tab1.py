import pandas as pd
import random
from dash import dcc, html, ALL, callback_context
import dash.dependencies as ddep
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import colorsys
from enum import Enum

from app import app

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

# Sample stock data
df = pd.DataFrame(
    {
        "Date": pd.date_range(start="2023-01-01", periods=100),
        "Company": ["Titou&Co"] * 50 + ["Company123"] * 50,
        "Value": list(range(50)) + list(range(50, 100)),
        "Open": [random.randint(10, 50) for _ in range(100)],
        "High": [random.randint(50, 100) for _ in range(100)],
        "Low": [random.randint(0, 10) for _ in range(100)],
        "Close": [random.randint(10, 50) for _ in range(100)],
    }
)


# Define Riskiness Levels with Corresponding Colors
class RiskinessLevel(Enum):
    STABLE = {"label": "Stable", "color": "green"}
    SLIGHTLY_UNSTABLE = {"label": "Slightly Unstable", "color": "orange"}
    HIGHLY_UNSTABLE = {"label": "Highly Unstable", "color": "red"}


# Updated riskiness mapping for companies
riskiness_mapping = {
    "Titou&Co": RiskinessLevel.STABLE,
    "Company123": RiskinessLevel.HIGHLY_UNSTABLE,
    "Company456": RiskinessLevel.SLIGHTLY_UNSTABLE,
}


# Generate random colors for companies
def generate_color_mapping(companies):
    random.seed(42)  # Set seed for reproducibility
    return {company: f"#{random.randint(0, 0xFFFFFF):06x}" for company in companies}


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


# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=5, num_std_dev=2):
    data["Moving Average"] = data["Value"].rolling(window=window).mean()
    data["Standard Deviation"] = data["Value"].rolling(window=window).std()
    data["Upper Band"] = data["Moving Average"] + (
        num_std_dev * data["Standard Deviation"]
    )
    data["Lower Band"] = data["Moving Average"] - (
        num_std_dev * data["Standard Deviation"]
    )
    return data


# Dynamically generate color mapping
color_mapping = generate_color_mapping(df["Company"].unique())

# Header for the layout
header = html.Div(
    [
        html.H1(
            "Bienvenue chez Ricou Bank!!",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
    ]
)

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

# Wrapping main content in cards
tab1_layout = html.Div(
    [
        bollinger_state_store,
        header,
        dbc.Card(
            dbc.CardBody(
                [
                    # Timeframe and chart type controls
                    html.H4("Chart Controls", className="card-title"),
                    html.Div(
                        [
                            # Timeframe Selection
                            html.Div(
                                [
                                    html.H5(
                                        "Select Timeframe",
                                        style={"marginRight": "20px"},
                                    ),
                                    dcc.DatePickerRange(
                                        id="date-picker",
                                        min_date_allowed=df["Date"].min(),
                                        max_date_allowed=df["Date"].max(),
                                        start_date=df["Date"].min(),
                                        end_date=df["Date"].max(),
                                        style={"marginRight": "20px"},
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
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
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
                                            {"label": company, "value": company}
                                            for company in sorted(
                                                df["Company"].unique()
                                            )
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
    style={"backgroundColor": theme["background"], "padding": "20px"},
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
    ],
    [ddep.State({"type": "bollinger-checkbox", "index": ALL}, "id")],
)
def update_chart(
    line_clicks,
    candlestick_clicks,
    selected_companies,
    start_date,
    end_date,
    bollinger_values,
    bollinger_state,
    checkbox_ids,
):
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

    # Filter data based on selected companies and timeframe
    filtered_df = df[
        (df["Company"].isin(selected_companies))
        & (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ]

    # Generate the appropriate chart
    fig = go.Figure()

    for company in selected_companies:
        company_data = filtered_df[filtered_df["Company"] == company]

        # Add Bollinger Bands to the graph with company-specific colors
        if company in companies_with_bollinger:
            company_data = calculate_bollinger_bands(company_data)

            # Add Bollinger Bands to the graph
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Upper Band"],
                    mode="lines",
                    name=f"{company} Upper Band",
                    line=dict(
                        color=color_mapping[company], dash="dot"
                    ),  # Use company color
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Moving Average"],
                    mode="lines",
                    name=f"{company} Moving Average",
                    line=dict(
                        color=color_mapping[company], dash="dash"
                    ),  # Use company color
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Lower Band"],
                    mode="lines",
                    name=f"{company} Lower Band",
                    line=dict(
                        color=color_mapping[company], dash="dot"
                    ),  # Use company color
                )
            )

        # Add the main line or candlestick chart
        if chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=company_data["Date"],
                    y=company_data["Value"],
                    mode="lines",
                    name=company,
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
                    name=company,
                    increasing_line_color=color_mapping[company],
                    decreasing_line_color=generate_complementary_color(
                        color_mapping[company], increase=False
                    ),  # Use complementary color
                )
            )

    fig.update_layout(
        title=graph_title,  # Dynamic title based on chart type
        xaxis_title="Time",
        yaxis_title="Stock Value (In USD)",
        template="plotly_white",
        title_x=0.5,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis_rangeslider_visible=False,  # Remove time slider for candlestick
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
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
    ],
    [ddep.State("bollinger-state-store", "data")],  # Add this state
)
def update_company_table(selected_companies, delete_clicks, bollinger_state):
    # Initialize selected_companies as an empty list if None
    if selected_companies is None:
        selected_companies = []

    # Initialize bollinger_state if None
    if bollinger_state is None:
        bollinger_state = {}

    ctx = callback_context

    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        try:
            triggered_dict = eval(triggered_id)
        except Exception:
            triggered_dict = {}

        if (
            isinstance(triggered_dict, dict)
            and triggered_dict.get("type") == "delete-button"
        ):
            company_to_remove = triggered_dict["index"]
            if company_to_remove in selected_companies:
                selected_companies.remove(company_to_remove)
                # Note: We don't remove from bollinger_state here so it remembers the choice

    # Update dropdown options to exclude selected companies
    all_options = [
        {"label": company, "value": company}
        for company in sorted(df["Company"].unique())
    ]

    # Generate the company table
    if selected_companies:
        table_header = html.Div(
            [
                html.Div(
                    "Riskiness",
                    style={
                        "width": "25%",
                        "fontWeight": "bold",
                        "textAlign": "left",
                        "padding": "8px",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
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
                    },
                ),
                html.Div(
                    "Bollinger",
                    style={
                        "width": "20%",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "padding": "8px",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
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
                        "overflow": "hidden",
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

            # Get riskiness info
            riskiness = riskiness_mapping.get(company, RiskinessLevel.STABLE)

            # Check if this company has Bollinger Bands enabled in the stored state
            is_bollinger_enabled = bollinger_state.get(company, False)

            table_rows.append(
                html.Div(
                    [
                        # Riskiness with colorful badge
                        html.Div(
                            html.Span(
                                riskiness.value["label"],
                                style={
                                    "backgroundColor": riskiness.value["color"],
                                    "color": "white",
                                    "padding": "3px 8px",
                                    "borderRadius": "12px",
                                    "fontSize": "0.85rem",
                                    "display": "inline-block",
                                },
                            ),
                            style={
                                "width": "25%",
                                "textAlign": "left",
                                "padding": "8px",
                            },
                        ),
                        # Company Name
                        html.Div(
                            company,
                            style={
                                "width": "35%",
                                "textAlign": "center",
                                "padding": "8px",
                                "fontWeight": "500",
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
                                "width": "20%",
                                "textAlign": "center",
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
                        "backgroundColor": row_bg_color,  # Alternating row colors
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
                html.Span(
                    "ℹ️",
                    style={"marginRight": "8px", "fontSize": "16px"},
                ),
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
