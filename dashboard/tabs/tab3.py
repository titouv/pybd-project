from dash import html, dcc, dash_table
import plotly.graph_objs as go
import dash.dependencies as ddep
import dash_extensions as de

from app import app, db  # Import these instances

tab3_layout = dcc.Tab(
    label="SQL",
    children=[
        html.H2("SQL Terminal (but better)"),
        dcc.Loading(
            id="sql-loading",
            type="circle",
            children=html.Div(
                id="sql-query-output",
                style={
                    "whiteSpace": "pre-line",
                    "overflowY": "auto",
                    "height": 500,
                    "border": "1px solid #ccc",
                    "padding": "10px",
                },
            ),
        ),
        de.Keyboard(
            dcc.Textarea(
                id="sql-query-input",
                style={"width": "100%", "height": "2em"},
                placeholder="Enter your SQL query here...",
            ),
            captureKeys=["Enter"],
            id="sql-query-key",
        ),
        html.Button("Run", id="sql-run-btn", n_clicks=0, style={"marginLeft": "10px"}),
        html.Button(
            "Clear Output", id="sql-clear-btn", n_clicks=0, style={"marginLeft": "10px"}
        ),
    ],
)


@app.callback(
    [
        ddep.Output("sql-query-output", "children"),
        ddep.Output("sql-query-input", "value"),
    ],
    [
        ddep.Input("sql-query-key", "n_keydowns"),
        ddep.Input("sql-run-btn", "n_clicks"),
        ddep.Input("sql-clear-btn", "n_clicks"),
    ],
    [
        ddep.State("sql-query-input", "value"),
        ddep.State("sql-query-output", "children"),
    ],
)
def execute_query(n_key, n_run, n_clear, query, history):
    from dash import callback_context

    if history is None:
        history = []
    ctx = callback_context

    # Determine which input triggered the callback
    if not ctx.triggered:
        return history, query

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "sql-clear-btn":
        return [], ""
    if (trigger == "sql-run-btn" or trigger == "sql-query-key") and query:
        try:
            result_df = db.df_query(query)
            if len(result_df) == 0:
                error_msg = str(db.logger.get_last_message())
                try:
                    error_msg = error_msg.split(") ")[1].split("\n\n")[0]
                except:
                    pass
                if error_msg != "df_query: " + query:
                    return (
                        history
                        + [
                            html.Span("error ", style={"color": "red"}),
                            html.Pre(error_msg),
                        ],
                        query,
                    )
            if not result_df.empty:
                table = dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in result_df.columns],
                    data=result_df.to_dict("records"),
                    page_size=10,
                    style_table={"overflowX": "auto"},
                )
                return history + [html.B(query), table, html.Br()], ""
            return (
                history + [html.Pre([html.B(query), result_df.to_string()]), html.Br()],
                "",
            )
        except Exception as e:
            return (
                history
                + [
                    html.Pre(str(e)),
                ],
                query,
            )
    # If nothing relevant triggered, return current state
    return history, query
