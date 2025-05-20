import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from nba_api.stats.endpoints import TeamGameLogs, PlayerGameLogs
from nba_api.stats.static import players

@st.cache_data
def build_feature_matrix(team_id: int, season: str, tags: dict) -> pd.DataFrame:
    games = TeamGameLogs(
        season_nullable=season,
        team_id_nullable=team_id,
        season_type_nullable="Regular Season"
    ).get_data_frames()[0][["GAME_ID", "WL"]]
    games["WIN"] = (games["WL"] == "W").astype(int)
    records = []
    for label, name in tags.items():
        pid = players.find_players_by_full_name(name)[0]["id"]
        df = PlayerGameLogs(
            season_nullable=season,
            player_id_nullable=pid,
            season_type_nullable="Regular Season"
        ).get_data_frames()[0][["GAME_ID", "PTS", "MIN"]]
        df.columns = ["GAME_ID", f"{label}_PTS", f"{label}_MIN"]
        records.append(df)
    from functools import reduce
    df_feat = reduce(lambda a, b: a.merge(b, on="GAME_ID"), records)
    df_feat = df_feat.merge(games[["GAME_ID", "WIN"]], on="GAME_ID")
    return df_feat.dropna()


def main():
    st.image("output.png", use_container_width=True)
    st.header("Win-Prediction Model Coefficients")
    team_id = 1610612745
    season = st.selectbox("Season (Model)", ["2024-25", "2023-24"])
    tags = {"SENGUN": "Alperen Sengun", "GREEN": "Jalen Green"}

    df = build_feature_matrix(team_id, season, tags)
    X = df.drop(columns=["GAME_ID", "WIN"]).astype(int)
    y = df["WIN"]
    model = LogisticRegressionCV(cv=5, max_iter=1000).fit(X, y)
    coef = pd.Series(model.coef_[0], index=X.columns)
    st.bar_chart(coef, use_container_width=True)


if __name__ == "__main__":
    main()