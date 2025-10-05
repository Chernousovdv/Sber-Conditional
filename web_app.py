import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import dash
import numpy as np
from dash import dcc, html


# ===============================
# Your plotting function (no fig.show!)
# ===============================
def plot_tsne_similarity_plotly(
    df: pd.DataFrame,
    metric,
    color_dict: dict = None,
    group_dict: dict = None,  # New argument for group mapping
    perplexity: int = 5,
    random_state: int = 42,
    title: str = "t-SNE Visualization of Time Series Similarity",
    annotate=True,
    s=8,
    legend=True,
    figsize=(800, 500),
    scaler=MinMaxScaler,
):
    """
    Interactive t-SNE visualization using Plotly.

    This function provides the same functionality as the matplotlib version
    but with interactive Plotly visualization.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex where each column
                           is a non-negative time series (e.g., a commodity price).
        metric (callable or str): The metric for calculating distance between
                                  time series. Can be a string like 'euclidean'
                                  or a callable function.
        color_dict (dict): Dictionary mapping column names to colors.
        group_dict (dict): Dictionary mapping column names to group names.
                          Used for hover information and legend.
        perplexity (int): A t-SNE parameter related to the number of nearest
                          neighbors considered for each point. Should be less than
                          the number of time series.
        random_state (int): Seed for the random number generator to ensure
                            reproducible results.
        title (str): The title for the plot.
        annotate (bool): For API consistency with matplotlib version. In Plotly,
                        annotations are always shown on hover.
        s (int): Marker size.
        legend (bool): Whether to show the legend.
        figsize (tuple): Figure size as (width, height) in pixels.
        scaler: Scaler to use for normalization.
    """
    # 1. Data Preparation
    series_data = df.T
    series_labels = series_data.index.tolist()

    if perplexity >= len(series_labels):
        print(f"Adjusting perplexity from {perplexity} to {len(series_labels) - 1}")
        perplexity = len(series_labels) - 1

    # 2. Normalization
    scaler = scaler()
    normalized_data = np.vstack(
        [
            scaler.fit_transform(series_data.iloc[i, :].values.reshape(-1, 1)).ravel()
            for i in range(series_data.shape[0])
        ]
    )

    # 3. t-SNE Execution
    tsne = TSNE(
        n_components=2,
        metric=metric,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(normalized_data)

    # 4. Prepare data for Plotly
    results_df = pd.DataFrame(
        {"x": tsne_results[:, 0], "y": tsne_results[:, 1], "series": series_labels}
    )

    # Add group information if group_dict is provided
    if group_dict is not None:
        results_df["group"] = [
            group_dict.get(label, "Unknown") for label in series_labels
        ]
    else:
        results_df["group"] = "No group"

    # 5. Create interactive plot
    if color_dict is not None:
        # Add colors and create color groups for proper legend
        results_df["color"] = [color_dict.get(label, "gray") for label in series_labels]
        results_df["color_group"] = results_df["color"]

        # Create custom hover text with series name and group
        results_df["hover_text"] = results_df.apply(
            lambda row: f"Series: {row['series']}<br>Group: {row['group']}", axis=1
        )

        fig = px.scatter(
            results_df,
            x="x",
            y="y",
            color="color_group",
            hover_name="hover_text",  # Use custom hover text
            title=title,
            size_max=s,
            color_discrete_map={color: color for color in results_df["color"].unique()},
        )

        # Update marker colors to use the actual colors from color_dict
        fig.update_traces(
            marker=dict(size=s, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )

        # Customize legend to show group names instead of colors
        if legend:
            fig.update_layout(showlegend=True)
            # Rename legend entries to show group names
            for trace in fig.data:
                color = trace.name
                # Find the group name for this color
                matching_rows = results_df[results_df["color"] == color]
                if not matching_rows.empty:
                    group_name = matching_rows["group"].iloc[0]
                    trace.name = f"{group_name}"
                else:
                    trace.name = f"{color} series"
        else:
            fig.update_layout(showlegend=False)

    else:
        # All points same color - still include group info in hover
        if group_dict is not None:
            results_df["hover_text"] = results_df.apply(
                lambda row: f"Series: {row['series']}<br>Group: {row['group']}", axis=1
            )
        else:
            results_df["hover_text"] = results_df["series"]

        fig = px.scatter(
            results_df,
            x="x",
            y="y",
            hover_name="hover_text",
            title=title,
            size_max=s,
        )

        fig.update_traces(
            marker=dict(
                size=s,
                opacity=0.7,
                color="blue",  # Default color
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            selector=dict(mode="markers"),
        )
        fig.update_layout(showlegend=False)

    # 6. Update layout
    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        width=figsize[0],
        height=figsize[1],
        title_x=0.5,  # Center the title
    )

    return fig  # return instead of fig.show()


# ===============================
# Load your data
# ===============================
df = pd.read_csv("/home/danilach/comb_df_v1.csv", index_col=0)
df.index = pd.to_datetime(df.index)


groups = {
    "Кукуруза": "Grains & Seeds",
    "Пшеница 1-го класса": "Grains & Seeds",
    "Пшеница 3-го класса": "Grains & Seeds",
    "Пшеница 4-го класса": "Grains & Seeds",
    "Пшеница 5-го класса": "Grains & Seeds",
    "Пшеница 12,5% FOB Ново, руб/т": "Grains & Seeds",
    "Подсолнечник": "Oils",
    "Соя": "Oils",
    "Рапс, руб./т": "Oils",
    "Подсолнечное масло наливом (мировые цены)": "Oils",
    "Бутилированное подсолнечное масло (рафинированное)": "Oils",
    "Подсолнечное масло (наливом) не бутилированное, нерафинированное ": "Oils",
    "Подсолнечный шрот ": "Oils",
    "Соевое масло": "Oils",
    "Соевый шрот": "Oils",
    "Рапсовое масло EU, руб/т": "Oils",
    "Кокосовое масло (USD) ": "Tropical Oils",
    "Кокосовое масло ": "Tropical Oils",
    "Пальмовое масло (USD)": "Tropical Oils",
    "Пальмовое масло ": "Tropical Oils",
    "Молоко сырое": "Livestock & Meat",
    "Мясо птицы бройлеров в живом весе ": "Livestock & Meat",
    "Мясо крупного рогатого скота в живом весе ": "Livestock & Meat",
    "Свинина в живом весе": "Livestock & Meat",
    "Баранина в живом весе": "Livestock & Meat",
    "Баранина в убойном весе ": "Livestock & Meat",
    "Яйцо товарное": "Livestock & Meat",
    "Колбасы сырокопченые": "Livestock & Meat",
    "Колбасы вареные": "Livestock & Meat",
    "Минтай б/г, Владивосток руб./кг": "Fish",
    "Минтай б/г, Китай C&F руб/т": "Fish",
    "Сельдь н/р, Владивосток руб./кг": "Fish",
    "Огурцы тепличные, руб./кг": "Vegetables",
    "Томаты тепличные, руб./кг": "Vegetables",
    "Сахар (средняя цена по России)": "Sugar",
    "Мука пшеничная, руб./т": "Grains & Seeds",
    "Какао-бобы (USD)": "Cacao",
    "Какао-бобы": "Cacao",
    "Табак (USD)": "Tobacco",
    "Табак": "Tobacco",
    "Карбамид (FOB Южный)": "dip",
    "Моноаммонийфосфат, MAP (FOB Балтика)": "dip",
    "Апатитовый концетрат (FOB Morocco)": "dip",
    "Аммиак (FOB Черное море)": "dip",
    "Аммиачная селитра (FOB Черное море)": "dip",
    "Хлорид калия (CFR Ю-В Азия)": "dip",
    "Капролактам импортный контракт (Тайвань и Ю. Корея) CFR Азия": "bump",
    "Метанол": "bump",
    "Бензол, CFR Япония": "bump",
    "Этилен, CFR Китай": "bump",
    "USDRUB": "Macro",
}

colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "violet",
    "olive",
    "cyan",
    "black",
    "yellow",
]

unique_values = list(set(groups.values()))
value_to_color = {value: colors[i] for i, value in enumerate(unique_values)}
color_dict = {key: value_to_color[value] for key, value in groups.items()}


fig = plot_tsne_similarity_plotly(
    df=df,
    metric="l2",
    color_dict=color_dict,
    perplexity=5,
    scaler=StandardScaler,
    group_dict=groups,
    title="T-SNE plot",
    legend=False,
)

# ===============================
# Dash app
# ===============================
app = dash.Dash(__name__)
app.layout = html.Div(
    [html.H1("t-SNE Visualization of Time Series Similarity"), dcc.Graph(figure=fig)]
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
