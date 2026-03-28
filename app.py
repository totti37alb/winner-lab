import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import re

# --- 0. ページ設定 ---
st.set_page_config(page_title="Totti's WINNER Lab", layout="wide")

# --- 1. CSS (視認性究極改善版) ---
st.markdown("""
<style>
    /* 全体：漆黒背景 */
    .stApp { background-color: #000000 !important; color: #ffffff !important; }

    /* ヘッダー */
    .header-box { text-align: center; padding: 25px; border-bottom: 2px solid #FF6700; background: #000000; margin-bottom: 20px; }
    .header-title { color: #FF6700 !important; font-size: 2.2rem !important; font-weight: 900; text-shadow: 0 0 10px rgba(255, 103, 0, 0.5); }

    /* Expanderを完全に黒く、文字を白く */
    div[data-testid="stExpander"] details { border: 1px solid #444 !important; background-color: #000000 !important; border-radius: 10px !important; margin-bottom: 10px !important; }
    div[data-testid="stExpander"] summary { background-color: #1a1a1a !important; color: #ffffff !important; padding: 12px !important; border-radius: 10px !important; }
    
    /* 文字色の強制固定（グレーアウト防止） */
    p, span, label, .stMarkdown, [data-testid="stMetricLabel"] { 
        color: #ffffff !important; 
        font-weight: bold !important; 
        opacity: 1.0 !important;
    }

    /* ボタン：ネオンオレンジ */
    div.stButton > button {
        background: linear-gradient(135deg, #FF6700 0%, #cc5200 100%) !important;
        color: #ffffff !important;
        border: none !important;
        width: 100% !important;
        height: 3.5em !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(255, 103, 0, 0.4) !important;
    }

    /* 予想数字：発光表示 */
    [data-testid="stMetricValue"] { 
        color: #FF6700 !important; 
        font-size: 2.5rem !important; 
        font-weight: 900 !important; 
        text-shadow: 0 0 15px rgba(255, 103, 0, 0.6);
    }
    
    /* 入力エリア */
    textarea { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #FF6700 !important; }

    /* タブ設定 */
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; }
    .stTabs [data-baseweb="tab"] { color: #888 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #FF6700 !important; border-bottom-color: #FF6700 !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. チームカラー定義 (J1・J2・J3 全60クラブ完全網羅版) ---
TEAM_COLORS = {
    # --- J1 ---
    "札幌": ["#E50012", "#000000"], "鹿島": ["#B11021", "#000000"], "浦和": ["#E60012", "#000000"],
    "柏": ["#FFF100", "#000000"], "FC東京": ["#0000FF", "#FF0000"], "東京V": ["#006400", "#DAA520"],
    "町田": ["#000080", "#FFD700"], "川崎F": ["#0099D9", "#000000"], "横浜FM": ["#0000FF", "#FF0000"],
    "湘南": ["#77FF00", "#0000FF"], "新潟": ["#FF6700", "#0047AB"], "磐田": ["#92B5D2", "#000000"],
    "名古屋": ["#D51621", "#FFAD00"], "京都": ["#8C0733", "#000000"], "G大阪": ["#0000FF", "#000000"],
    "C大阪": ["#E3007F", "#000080"], "神戸": ["#86001E", "#000000"], "広島": ["#502C83", "#000000"],
    "福岡": ["#002E5D", "#A4B1D7"], "鳥栖": ["#17E0FD", "#FF1493"],
    # --- J2 ---
    "仙台": ["#FFD700", "#0000FF"], "秋田": ["#0055FF", "#FFD700"], "山形": ["#0000FF", "#FFFFFF"],
    "いわき": ["#FF4500", "#000080"], "水戸": ["#0000FF", "#FF0000"], "栃木": ["#FFFF00", "#0000FF"],
    "群馬": ["#003366", "#FFFF00"], "千葉": ["#FFF100", "#009944"], "横浜FC": ["#00AEEF", "#000000"],
    "甲府": ["#0000FF", "#FF0000"], "清水": ["#FF8C00", "#000080"], "藤枝": ["#800080", "#FFFFFF"],
    "岡山": ["#8B0000", "#000000"], "山口": ["#FF4500", "#000000"], "徳島": ["#000080", "#FFFFFF"],
    "愛媛": ["#FF8C00", "#006400"], "長崎": ["#0055FF", "#FF8C00"], "熊本": ["#FF0000", "#000000"],
    "大分": ["#0000FF", "#FFFF00"], "鹿児島": ["#000080", "#FFFFFF"],
    # --- J3 ---
    "八戸": ["#008000", "#FFFFFF"], "岩手": ["#FFFFFF", "#000000"], "福島": ["#FF0000", "#FFFF00"],
    "大宮": ["#FF6600", "#000080"], "松本": ["#006400", "#FFFFFF"], "長野": ["#FF8C00", "#000080"],
    "富山": ["#0000FF", "#FF0000"], "金沢": ["#FF0000", "#FFFF00"], "沼津": ["#0000FF", "#FFFFFF"],
    "岐阜": ["#006400", "#FFFFFF"], "奈良": ["#000080", "#FFFFFF"], "FC大阪": ["#ADD8E6", "#000080"],
    "讃岐": ["#ADD8E6", "#000080"], "今治": ["#007BFF", "#FFF100"], "北九州": ["#FFFF00", "#FF0000"],
    "宮崎": ["#FFFFFF", "#000000"], "琉球": ["#8B0000", "#DAA520"], "鳥取": ["#00FF00", "#FFFFFF"],
    "滋賀": ["#0000FF", "#FFFFFF"], "相模原": ["#006400", "#000000"]
}
LIGHT_BG = ["#FFF100", "#FFFF33", "#FFFFFF"]

# --- 3. ロジック (Jリーグ特化：乱打戦抑制モード) ---
@st.cache_data
def load_data():
    try:
        tmp = pd.read_csv("teams.csv")
        tmp.columns = tmp.columns.str.strip()
        return tmp
    except: return pd.DataFrame()

def predict_score(h_at, h_df, a_at, a_df):
    # 係数を 0.08 から 0.11 に引き上げ（少し攻撃的に戻す）
    mu_h = (h_at * 0.10) / (a_df * 0.10)
    mu_a = (a_at * 0.10) / (h_df * 0.10)
    
    probs = []
    for h in range(4):
        for a in range(4):
            p = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
            probs.append({"score": f"{h}-{a}", "prob": p})
            
    # その他勝率の補正を 0.4 から 0.8 に緩和（適度な大勝確率を許容）
    p_h_other = sum(poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a) for h in range(4, 10) for a in range(h)) * 0.8
    p_a_other = sum(poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a) for a in range(a, 10) for h in range(a)) * 0.8
    
    probs.append({"score": "その他(H勝)", "prob": p_h_other})
    probs.append({"score": "その他(A勝)", "prob": p_a_other})
    
    return sorted(probs, key=lambda x: x['prob'], reverse=True)

# --- 4. メイン表示 ---
df = load_data()
st.markdown('<div class="header-box"><h1 class="header-title">Totti\'s WINNER Lab</h1></div>', unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["🎯 期待値スキャン", "🧠 結果学習", "📊 リーグ戦力表"])

with t1:
    txt = st.text_area("Jリーグ公式サイトの今週の日程・結果ページのコピペ、または「新潟 浦和」でOK", height=150)
    if st.button("期待値をスキャン！"):
        if not txt:
            st.warning("データを入力してください")
        elif df.empty:
            st.error("teams.csvを読み込めません")
        else:
            # 1. 記号の正規化（全角VS、スペース、タブ等を整理）
            processed_txt = txt.replace('　', ' ').replace('ＶＳ', ' ').replace('vs', ' ').replace('VS', ' ').replace('\t', ' ')
            lines = processed_txt.split('\n')
            
            clean_matches = []
            team_list = df["チーム名"].tolist()
            # チーム名の長い順にソート（「横浜FM」を「横浜」で誤判定しないため）
            sorted_teams = sorted(team_list, key=len, reverse=True)

            for line in lines:
                if not line.strip(): continue
                
                hits = []
                # 2. その行に含まれるチーム名とその出現位置を記録
                for t_name in sorted_teams:
                    if t_name in line:
                        # 同じチーム名が1行に複数あっても対応できるようfindall的に探す
                        start_pos = 0
                        while True:
                            pos = line.find(t_name, start_pos)
                            if pos == -1: break
                            hits.append((pos, t_name))
                            start_pos = pos + len(t_name)
                
                # 3. 出現位置順に並べて、重複（長い名前に含まれる短い名前）を排除
                hits.sort()
                final_hit_names = []
                last_end = -1
                for pos, name in hits:
                    if pos >= last_end: # 前のチーム名と被ってなければ採用
                        final_hit_names.append(name)
                        last_end = pos + len(name)
                
                # 4. 1行に2つ以上あればペアにする
                # 「神戸vs広島の試合詳細」→ [神戸, 広島] → (神戸, 広島)
                # 「新潟 浦和」→ [新潟, 浦和] → (新潟, 浦和)
                if len(final_hit_names) >= 2:
                    # 最初の2つをHome, Awayとする
                    h_team, a_team = final_hit_names[0], final_hit_names[1]
                    if h_team != a_team:
                        clean_matches.append((h_team, a_team))

            # 全体の重複カードを削除
            clean_matches = list(dict.fromkeys(clean_matches))

            if not clean_matches:
                st.error("対戦カードが判別できません。チーム名が2つ入っているか確認してください。")
            else:
                st.success(f"🎯 {len(clean_matches)}試合を検出！")
                for h, a in clean_matches:
                    th = df[df["チーム名"]==h].iloc[0]
                    ta = df[df["チーム名"]==a].iloc[0]
                    res = predict_score(th["攻撃力"], th["守備力"], ta["攻撃力"], ta["守備力"])
                    
                    with st.expander(f"🏟️ {h} vs {a}", expanded=True):
                        c = st.columns(3)
                        for idx, r in enumerate(res[:3]):
                            c[idx].metric(f"予想 {idx+1}", r["score"], f"{r['prob']:.1%}")

                            
with t2:
    st.markdown("### 🧠 試合結果から戦力調整")
    st_res = st.text_area("結果入力 (例: 新潟 2 - 1 浦和)", height=150)
    lr = st.slider("学習の強さ", 0.01, 0.20, 0.05)
    if st.button("学習を実行"):
        matches = re.findall(r'([A-Za-z一-龠ぁ-ヶー]+)\s+(\d+)\s*-\s*(\d+)\s+([A-Za-z一-龠ぁ-ヶー]+)', st_res)
        if not matches:
            st.error("試合結果が見つかりません。形式を確認してください。")
        else:
            for h_n, h_s, a_s, a_n in matches:
                if h_n in df["チーム名"].values and a_n in df["チーム名"].values:
                    h_i, a_i = df[df["チーム名"]==h_n].index[0], df[df["チーム名"]==a_n].index[0]
                    # 基準を1.0点に厳格化（乱打戦予想を抑制）
                    df.at[h_i, "攻撃力"] += (int(h_s) - 1.0) * lr
                    df.at[a_i, "攻撃力"] += (int(a_s) - 1.0) * lr
                    df.at[h_i, "守備力"] -= (int(a_s) - 1.0) * lr
                    df.at[a_i, "守備力"] -= (int(h_s) - 1.0) * lr
            df["攻撃力"] = df["攻撃力"].clip(1.0, 25.0).round(2)
            df["守備力"] = df["守備力"].clip(1.0, 25.0).round(2)
            df.to_csv("teams.csv", index=False)
            st.success("teams.csvを学習・更新しました！")

with t3:
    st.markdown("### 📊 ブロック別・戦力ステータス")
    if not df.empty:
        # カテゴリ名をCSVの「カテゴリ」列と完全に一致させる
        cats = ["J1-EAST", "J1-WEST", "J23-EAST-A", "J23-EAST-B", "J23-WEST-A", "J23-WEST-B"]
        ctabs = st.tabs(["J1E", "J1W", "EA", "EB", "WA", "WB"])
        
        for i, cat in enumerate(cats):
            with ctabs[i]:
                # 指定したカテゴリのチームを抽出して攻撃力順に並べる
                tdf = df[df["カテゴリ"] == cat].sort_values("攻撃力", ascending=False)
                
                if tdf.empty:
                    st.write(f"{cat} のデータはありません")
                else:
                    for _, r in tdf.iterrows():
                        tn = r["チーム名"]
                        cols = TEAM_COLORS.get(tn, ["#333333", "#ffffff"])
                        bg = f"linear-gradient(135deg, {cols[0]} 70%, {cols[1]} 70%)"
                        # 背景が明るい場合は文字を黒にする判定
                        tc = "#000000" if cols[0] in LIGHT_BG else "#ffffff"
                        
                        st.markdown(f'''
                            <div style="background:{bg}; color:{tc}; padding:15px; border-radius:12px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(255,255,255,0.1);">
                                <b style="font-size:1.3rem;">{tn}</b>
                                <span style="background:rgba(0,0,0,0.7); padding:8px 15px; border-radius:8px; color:white; font-size:0.9rem; font-weight:900;">
                                    攻:{r["攻撃力"]:.1f} 守:{r["守備力"]:.1f}
                                </span>
                            </div>
                        ''', unsafe_allow_html=True)
