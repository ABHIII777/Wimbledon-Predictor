import streamlit as st
import pandas as pd
import joblib
import time
import glob
import plotly.graph_objects as go
from difflib import get_close_matches

model = joblib.load("model/model.pkl")
player_stats = joblib.load("model/player_stats.pkl")
scaler = joblib.load("model/scaler.pkl")
X_columns = joblib.load("model/feature_columns.pkl")

st.set_page_config(page_title="🎾 Wimbledon Winner Predictor", layout="wide")
st.markdown("<h1 style='text-align:center;'>🎾 Wimbledon Winner Predictor</h1>", unsafe_allow_html=True)

col1, mid, col2 = st.columns([4, 1, 4])
players = sorted(player_stats.keys())

with col1:
    player1 = st.selectbox("🎯 Select Player 1", players, key="p1")
with col2:
    player2 = st.selectbox("🔥 Select Player 2", players, key="p2")

surface = st.selectbox("🌍 Select Surface Type", ["Grass", "Clay", "Hard", "Carpet"], index=0)

st.markdown(
    f"""
    <div style='text-align:center; margin-top: 20px; font-size:24px;'>
        <b>{player1}</b> 🆚 <b>{player2}</b> <br>
        <small>Surface: <b>{surface}</b></small>
    </div>
    """,
    unsafe_allow_html=True
)

def predict_match(player1, player2, surface):
    p1 = player_stats.get(player1, {})
    p2 = player_stats.get(player2, {})

    diff = {}
    for key in p1.keys():
        if isinstance(p1[key], (int, float)) and isinstance(p2.get(key, 0), (int, float)):
            diff[f"diff_pref_{key}"] = p1[key] - p2[key]

    diff["elo_diff"] = p1.get("elo", 1500) - p2.get("elo", 1500)
    diff["recent_wr_diff"] = p1.get("recent_wr", 0.5) - p2.get("recent_wr", 0.5)

    for surf in ["grass", "clay", "hard", "carpet"]:
        diff[f"surface_{surf.lower()}"] = 1.0 if surface.lower() == surf else 0.0

    Xp = pd.DataFrame([diff])
    for col in X_columns:
        if col not in Xp.columns:
            Xp[col] = 0
    Xp = Xp[X_columns]

    Xp_scaled = scaler.transform(Xp)
    prob = model.predict_proba(Xp_scaled)[0][1]
    return round(prob * 100, 2), round((1 - prob) * 100, 2)

predict_btn = st.button("⚡ Predict", use_container_width=True)

if predict_btn:
    with st.spinner("🎾 Analyzing players' stats and ELO ratings..."):
        time.sleep(1.8)
        prob1, prob2 = predict_match(player1, player2, surface)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style='text-align:center; background-color:#E3F2FD; padding:20px; border-radius:15px;'>
                <h3>{player1}</h3>
                <h2 style='color:#1976D2;'>{prob1}%</h2>
                <p>Winning Probability on {surface}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style='text-align:center; background-color:#FCE4EC; padding:20px; border-radius:15px;'>
                <h3>{player2}</h3>
                <h2 style='color:#C2185B;'>{prob2}%</h2>
                <p>Winning Probability on {surface}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<hr><h2 style='text-align:center;'>📊 Player Form Trend (Past 10 Years)</h2>", unsafe_allow_html=True)


@st.cache_data
def load_past_data():
    """Load ATP matches from the last 10 years."""
    files = sorted(glob.glob("data/raw/atp_matches_*.csv"))
    dfs = []
    for f in files:
        year = f.split("_")[-1].split(".")[0]
        if year.isdigit() and int(year) >= 2015:
            df = pd.read_csv(f)
            df["year"] = int(year)
            dfs.append(df)
    if not dfs:
        raise ValueError("No ATP match CSVs found in data/raw/")
    return pd.concat(dfs, ignore_index=True)

def normalize_name(name: str) -> str:
    """Simplify player names to improve matching."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower().replace(".", "").replace("-", " ")
    parts = name.split()
    if len(parts) == 2:
        return f"{parts[0][0]} {parts[1]}"
    return name

try:
    matches = load_past_data()
    matches.columns = [c.lower() for c in matches.columns]

    winner_col = next(c for c in matches.columns if "winner" in c and "name" in c)
    loser_col = next(c for c in matches.columns if "loser" in c and "name" in c)

    matches["winner_norm"] = matches[winner_col].apply(normalize_name)
    matches["loser_norm"] = matches[loser_col].apply(normalize_name)

    player1_norm = normalize_name(player1)
    player2_norm = normalize_name(player2)

    all_names = pd.concat([matches["winner_norm"], matches["loser_norm"]]).unique()
    p1_match = get_close_matches(player1_norm, all_names, n=1, cutoff=0.5)
    p2_match = get_close_matches(player2_norm, all_names, n=1, cutoff=0.5)

    if not p1_match or not p2_match:
        st.warning("⚠️ Player names not found in historical data.")
    else:
        p1_match = p1_match[0]
        p2_match = p2_match[0]

        def yearly_winrate(norm_name):
            df_p = matches[(matches["winner_norm"] == norm_name) | (matches["loser_norm"] == norm_name)]
            if df_p.empty:
                return pd.DataFrame(columns=["year", "winrate"])
            wins = df_p[df_p["winner_norm"] == norm_name].groupby("year").size()
            total = df_p.groupby("year").size()
            wr = (wins / total).fillna(0)
            return wr.reset_index(name="winrate")

        df1 = yearly_winrate(p1_match)
        df2 = yearly_winrate(p2_match)

        if df1.empty and df2.empty:
            st.warning("😕 No historical matches found for these players.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df1["year"], y=df1["winrate"], mode="lines+markers",
                                     name=player1, line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df2["year"], y=df2["winrate"], mode="lines+markers",
                                     name=player2, line=dict(color="red")))
            fig.update_layout(
                title="Win Rate Over the Past 10 Years",
                xaxis_title="Year",
                yaxis_title="Win Rate",
                yaxis=dict(tickformat=".0%"),
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Could not plot player trends: {e}")


st.markdown("<br><hr><center>💻 Built with Streamlit + XGBoost + ELO Ratings</center>", unsafe_allow_html=True)
