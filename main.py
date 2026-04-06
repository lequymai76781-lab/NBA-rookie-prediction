from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="NBA新秀生涯潜力预测 API", version="1.0")

# ====================== CORS（支持 GitHub Pages 前端）======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",  # 本地前端地址
        "https://lequymai76781-lab.github.io",  # 你的GitHub Pages
        "https://wuhw.fun"  # 你买的自定义域名
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== 全局加载 ======================
model = None
le = None
feature_cols = None
preprocessor = None
rookie_df = None
veteran_df = None


@app.on_event("startup")
async def load_all_resources():
    global model, le, feature_cols, preprocessor, rookie_df, veteran_df

    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "le.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))

    rookie_df = pd.read_csv(os.path.join(BASE_DIR, "current_rookies.csv"))
    veteran_df = pd.read_csv(os.path.join(BASE_DIR, "historical_veterans.csv"))

    # 清理球员名（和前端保持一致）
    rookie_df['player_name'] = rookie_df['player_name'].astype(str).str.strip()
    print("🚀 FastAPI 资源加载完成")


# ====================== 辅助函数（直接复制自你的 Streamlit）=====================
def clean_search_key(key: str) -> str:
    if not isinstance(key, str):
        return ""
    return key.lower().strip().replace(" ", "").replace("·", "")


def calculate_similar_players(rookie_row, top_n=5):
    player_name_lower = str(rookie_row.get('player_name', '')).lower()

    # 文班亚马专用模板
    if 'wembanyama' in player_name_lower:
        return [
            {"name": "Tim Duncan", "type": "超长巅峰型", "score": 21.1, "career_length": 19, "reb": 11.0, "ast": 3.2,
             "stl": 0.7, "blk": 2.2, "report": "历史第一大前锋，19年生涯，5冠3FMVP2MVP，攻防一体。"},
            {"name": "Hakeem Olajuwon", "type": "超长巅峰型", "score": 20.6, "career_length": 18, "reb": 11.1,
             "ast": 2.5, "stl": 1.7, "blk": 3.1, "report": "梦幻脚步，历史盖帽王，2冠2DPOY1MVP。"},
            {"name": "David Robinson", "type": "超长巅峰型", "score": 24.3, "career_length": 14, "reb": 10.6,
             "ast": 2.5, "stl": 1.4, "blk": 3.0, "report": "海军上将，MVP+DPOY，14年生涯，2冠。"},
            {"name": "Kevin Durant", "type": "超长巅峰型", "score": 20.3, "career_length": 16, "reb": 6.4, "ast": 3.0,
             "stl": 1.1, "blk": 1.1, "report": "历史级得分手，2冠2FMVP1MVP。"},
            {"name": "Anthony Davis", "type": "超长巅峰型", "score": 20.8, "career_length": 12, "reb": 10.2, "ast": 2.3,
             "stl": 1.3, "blk": 2.3, "report": "全能内线，1冠，多次最佳阵容。"}
        ]

    # 普通相似度计算
    valid_feature_cols = [col for col in feature_cols if col in veteran_df.columns]
    if not valid_feature_cols:
        return []

    rookie_features = rookie_row[valid_feature_cols].fillna(0).values.reshape(1, -1)
    veteran_features = veteran_df[valid_feature_cols].fillna(0).values

    scaler = StandardScaler()
    all_features = np.vstack([rookie_features, veteran_features])
    scaler.fit(all_features)
    rookie_scaled = scaler.transform(rookie_features)
    veteran_scaled = scaler.transform(veteran_features)

    cos_sim = cosine_similarity(rookie_scaled, veteran_scaled)[0]
    similarity_score = (cos_sim + 1) * 50

    veteran_df_temp = veteran_df.copy()
    veteran_df_temp['similarity'] = similarity_score
    top_veterans = veteran_df_temp.nlargest(top_n, 'similarity')

    similar = []
    for _, row in top_veterans.iterrows():
        similar.append({
            "name": row.get('player_name', '未知'),
            "type": row.get('career_type', '未知'),
            "score": float(row.get('rookie_avg_score', 0)),
            "career_length": int(row.get('生涯长度', 0)),
            "reb": float(row.get('reb', 0)),
            "ast": float(row.get('ast', 0)),
            "stl": float(row.get('stl', 0)),
            "blk": float(row.get('blk', 0)),
            "similarity": round(float(row['similarity']), 1)
        })
    return similar


# ====================== 请求模型 ======================
class PredictRequest(BaseModel):
    player_name: str


# ====================== API 接口 ======================
@app.get("/api/rookies")
async def get_rookies():
    """返回所有新秀列表（前端搜索用）"""
    return rookie_df.to_dict(orient="records")


@app.get("/api/historical_veterans")
async def get_historical_veterans():
    """返回历史老将数据（相似度计算用）"""
    return veteran_df.to_dict(orient="records")


@app.post("/api/predict")
async def predict_career(request: PredictRequest):
    """核心预测接口"""
    player = rookie_df[rookie_df['player_name'].str.strip() == request.player_name.strip()]
    if player.empty:
        raise HTTPException(status_code=404, detail="未找到该球员")

    rookie_row = player.iloc[0].to_dict()

    # 1. 模型预测
    X = pd.DataFrame([rookie_row])[feature_cols].fillna(0)
    X_processed = preprocessor.transform(X)
    proba = model.predict_proba(X_processed)[0]
    pred_idx = np.argmax(proba)
    pred_type = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    # 2. 相似球员
    similar_players = calculate_similar_players(rookie_row)

    # 3. 综合潜力评分（和前端保持一致）
    potential_score = 92 if 'wembanyama' in request.player_name.lower() else \
        88 if 'banchero' in request.player_name.lower() else \
            85 if 'holmgren' in request.player_name.lower() else 80

    result = {
        "player_name": rookie_row.get("player_name"),
        "player_name_cn": rookie_row.get("player_name_cn", rookie_row.get("player_name")),
        "pred_type": pred_type,
        "confidence": confidence,
        "potential_score": potential_score,
        "sub_type": rookie_row.get("二级分类", "球队基石型"),
        "player_position": rookie_row.get("球员定位", "未知"),
        "newbie_age": int(rookie_row.get("新秀年龄", 0)),
        "early_avg_score": float(rookie_row.get("早期平均得分", 0)),
        "durability": float(rookie_row.get("早期出勤率", 0)) * 100,
        "injury_risk": float(rookie_row.get("伤病风险评分IRS", 50)),
        "similar_players": similar_players,
        "career_length": int(rookie_row.get("生涯长度", 0))
    }
    return result


# ====================== 健康检查 ======================
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "NBA新秀潜力预测 API 运行正常"}

# ====================== 运行 ======================
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload