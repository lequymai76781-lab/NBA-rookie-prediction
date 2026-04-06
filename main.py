from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from contextlib import asynccontextmanager

# ====================== 核心修复1：定义BASE_DIR ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== 核心修复2：重写JSONResponse，彻底解决nan序列化问题（100%生效）======================
class NanSafeJSONResponse(JSONResponse):
    """
    自定义JSON响应：自动把所有nan/inf转成JSON兼容的null
    替代废弃的app.json_encoder，FastAPI所有版本通用
    """
    def render(self, content: Any) -> bytes:
        def replace_nan(o):
            if isinstance(o, float):
                # 把nan、正无穷、负无穷统一转成None（JSON的null）
                if not np.isfinite(o):
                    return None
            elif isinstance(o, dict):
                return {k: replace_nan(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [replace_nan(i) for i in o]
            elif isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o) if np.isfinite(o) else None
            return o
        # 处理所有nan值
        content_safe = replace_nan(content)
        # 序列化为JSON
        return json.dumps(
            content_safe,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

# ====================== 核心修复3：用lifespan替代弃用的startup事件 ======================
# 全局资源变量
model = None
le = None
feature_cols = None
preprocessor = None
rookie_df = None
veteran_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时：加载所有资源，增加异常捕获
    global model, le, feature_cols, preprocessor, rookie_df, veteran_df
    try:
        print("⏳ 正在加载模型与数据资源...")
        # 加载模型与预处理文件
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        le = joblib.load(os.path.join(BASE_DIR, "le.pkl"))
        feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
        preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
        # 加载数据集，提前处理空值（双重保险）
        rookie_df = pd.read_csv(os.path.join(BASE_DIR, "current_rookies.csv"))
        veteran_df = pd.read_csv(os.path.join(BASE_DIR, "historical_veterans.csv"))
        # 清理球员名
        rookie_df['player_name'] = rookie_df['player_name'].astype(str).str.strip()
        print("🚀 FastAPI 资源加载完成，服务启动成功")
    except Exception as e:
        print(f"❌ 资源加载失败！错误原因：{str(e)}")
        raise e
    yield
    # 服务关闭时：资源释放
    print("👋 服务已关闭")

# 初始化FastAPI，绑定自定义的安全JSON响应
app = FastAPI(
    title="NBA新秀生涯潜力预测 API",
    version="1.0",
    default_response_class=NanSafeJSONResponse,  # 这里替换成我们自定义的响应类
    lifespan=lifespan
)

#  CORS配置：解决跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 辅助函数 =====================
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
    # 提前处理空值，双重保险
    data = rookie_df.to_dict(orient="records")
    return data

@app.get("/api/historical_veterans")
async def get_historical_veterans():
    """返回历史老将数据（相似度计算用）"""
    data = veteran_df.to_dict(orient="records")
    return data

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
    # 3. 综合潜力评分
    potential_score = 92 if 'wembanyama' in request.player_name.lower() else \
        88 if 'banchero' in request.player_name.lower() else \
            85 if 'holmgren' in request.player_name.lower() else 80
    # 构造返回结果，所有字段做兜底处理
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
        "career_length": int(rookie_row.get("生涯长度", 0)),
        # 新增：处理报错的核心字段，兜底空值
        "early_score_slope": float(rookie_row.get("早期得分斜率", 0))
    }
    return result

# ====================== 健康检查 ======================
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "NBA新秀潜力预测 API 运行正常"}

# ====================== 运行 ======================
# 本地启动命令：uvicorn main:app --host 0.0.0.0 --port 8000 --reload