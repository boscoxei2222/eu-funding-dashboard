import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import networkx as nx
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# 数据加载与预处理
# 读取主项目文件（Excel 格式）
df_projects = pd.read_excel("/Users/lirenxie/Desktop/horizon_dashboard/data/project.xlsx", usecols=["id", "ecMaxContribution", "fundingScheme", "topics"])

# 读取机构文件（Excel 格式）
df_orgs = pd.read_excel("/Users/lirenxie/Desktop/horizon_dashboard/data/organization.xlsx")
df_merged = pd.merge(df_orgs, df_projects, left_on="projectID", right_on="id", how="left")
df_merged[["lat", "lon"]] = df_merged["geolocation"].str.split(",", expand=True).astype(float)

# 按机构聚合数据
df_institutions = df_merged.groupby(["organisationID", "name", "country", "lat", "lon"]).agg(
    total_funding=("ecContribution", "sum"),
    project_count=("projectID", "nunique")
).reset_index()

# 构建合作网络边列表
edges = []
for project_id in df_merged["projectID"].unique():
    orgs = df_merged[df_merged["projectID"] == project_id]["organisationID"].tolist()
    for pair in combinations(orgs, 2):
        edges.append((pair[0], pair[1]))
df_edges = pd.DataFrame(edges, columns=["source", "target"])

# 计算 PageRank 影响力
G = nx.from_pandas_edgelist(df_edges, "source", "target")
pr = nx.pagerank(G)
df_institutions["pagerank"] = df_institutions["organisationID"].map(pr)

# 归一化与聚类（处理 NaN）
scaler = MinMaxScaler()
df_institutions[["norm_funding", "norm_pagerank"]] = scaler.fit_transform(
    df_institutions[["total_funding", "pagerank"]].fillna(0)
)
df_institutions["influence"] = 0.6 * df_institutions["norm_funding"] + 0.4 * df_institutions["norm_pagerank"]

# 仅对非 NaN 的行聚类
valid_influence = df_institutions["influence"].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
df_institutions.loc[valid_influence.index, "cluster"] = kmeans.fit_predict(valid_influence.values.reshape(-1, 1))

# 初始化 Dash 应用
app = dash.Dash(__name__)

# 定义布局
app.layout = html.Div([
    dcc.Input(id="search-input", placeholder="Search by institution name..."),
    dcc.Graph(id="map"),
    html.Div(id="institution-panels")
])

# 回调：更新地图
@app.callback(
    Output("map", "figure"),
    Input("search-input", "value")
)
def update_map(search_term):
    filtered = df_institutions.copy()
    if search_term:
        filtered = filtered[filtered["name"].str.contains(search_term, case=False)]
    fig = px.scatter_mapbox(
        filtered,
        lat="lat",
        lon="lon",
        hover_name="name",
        hover_data=["country", "total_funding"],
        size="total_funding",
        color="cluster",
        zoom=3
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig

# 回调：显示机构信息面板
@app.callback(
    Output("institution-panels", "children"),
    Input("map", "clickData")
)
def update_panels(click_data):
    if not click_data:
        return []
    institution_name = click_data["points"][0]["hovertext"]
    institution = df_institutions[df_institutions["name"] == institution_name].iloc[0]
    projects = df_merged[df_merged["organisationID"] == institution["organisationID"]]
    
    # 生成项目列表
    project_list = []
    for _, row in projects.iterrows():
        role = "Coordinator" if row["role"] == "coordinator" else "Partner"
        project_list.append(html.Li([
            html.Strong(row["projectAcronym"]),
            f" ({role}) - {row['topics']}"
        ]))
    
    # 生成合作网络图
    def generate_network(institution_id):
        projects = df_merged[df_merged["organisationID"] == institution_id]["projectID"]
        related_orgs = df_merged[df_merged["projectID"].isin(projects)]["organisationID"].unique()
        G = nx.Graph()
        for org in related_orgs:
            G.add_node(org, label=df_institutions[df_institutions["organisationID"] == org]["name"].values[0])
        for project in projects:
            orgs_in_project = df_merged[df_merged["projectID"] == project]["organisationID"].tolist()
            for pair in combinations(orgs_in_project, 2):
                if G.has_edge(pair[0], pair[1]):
                    G[pair[0]][pair[1]]["weight"] += 1
                else:
                    G.add_edge(pair[0], pair[1], weight=1)
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_labels = [G.nodes[node]["label"] for node in G.nodes()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888")))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=node_labels,
            marker=dict(size=10, color="blue")
        ))
        return dcc.Graph(figure=fig)
    
    network_fig = generate_network(institution["organisationID"])
    
    return html.Div([
        html.H3(institution["name"]),
        html.P(f"Country: {institution['country']}"),
        html.P(f"Total EU Funding: €{institution['total_funding']:,.0f}"),
        html.P(f"Influence Score: {institution['influence']:.2f}"),
        html.H4("Projects Involved"),
        html.Ul(project_list),
        html.H4("Collaboration Network"),
        network_fig
    ])

if __name__ == "__main__":
    app.run_server(debug=True)