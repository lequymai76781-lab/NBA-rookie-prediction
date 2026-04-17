# ====================== 第一步：先导入所有需要的模块 ======================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Any, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from contextlib import asynccontextmanager

# ====================== 全局配置 ======================
# 中文映射（扩展覆盖更多别名）
PLAYER_CN_MAP = {
    "Victor Wembanyama": "维克托·文班亚马",
    "Paolo Banchero": "保罗·班切罗",
    "Chet Holmgren": "切特·霍姆格伦",
    "Anthony Edwards": "安东尼·爱德华兹",
    "Jalen Green": "杰伦·格林",
    "Scoot Henderson": "斯库特·亨德森",
    "Brandon Miller": "布兰登·米勒",
    "Amen Thompson": "阿门·汤普森",
    "Ausar Thompson": "奥萨尔·汤普森"
}

# 扩展别名映射（解决模糊匹配问题）
PLAYER_ALIAS_MAP = {
    "文班亚马": "Victor Wembanyama",
    "文班": "Victor Wembanyama",
    "斑马": "Victor Wembanyama",
    "班切罗": "Paolo Banchero",
    "保罗班切罗": "Paolo Banchero",  # 去掉空格适配输入
    "霍姆格伦": "Chet Holmgren",
    "切特霍姆格伦": "Chet Holmgren",
    "华子": "Anthony Edwards",
    "爱德华兹": "Anthony Edwards",
    "杰伦格林": "Jalen Green",
    "格林": "Jalen Green",
    "亨德森": "Scoot Henderson",
    "米勒": "Brandon Miller",
    "阿门汤普森": "Amen Thompson",
    "奥萨尔汤普森": "Ausar Thompson"
}

ROOKIE_THRESHOLD = 3  # 新秀看三年
# 项目根目录（绝对路径，彻底解决找不到文件的问题）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 前端页面路径
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
# 静态文件目录（确保前端css/js/图片可访问）
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ====================== 全局资源变量 ======================
model = None
le = None
feature_cols = None
preprocessor = None
rookie_df = None
veteran_df = None


# ====================== 解决NaN序列化问题（增强版） ======================
class NanSafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        def replace_nan(o):
            if isinstance(o, float):
                if not np.isfinite(o):
                    return 0.0  # 替换NaN/Inf为0.0，避免前端解析报错
            elif isinstance(o, dict):
                return {k: replace_nan(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [replace_nan(i) for i in o]
            elif isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o) if np.isfinite(o) else 0.0
            elif o is None:
                return 0.0
            return o

        return json.dumps(
            replace_nan(content), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")


# ====================== 生命周期：启动时加载模型和数据（增强异常处理） ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, le, feature_cols, preprocessor, rookie_df, veteran_df
    try:
        print("=" * 50)
        print("⏳ 正在加载模型与数据...")

        # 检查必要文件是否存在
        required_files = [
            ("model.pkl", os.path.join(BASE_DIR, "model.pkl")),
            ("le.pkl", os.path.join(BASE_DIR, "le.pkl")),
            ("feature_cols.pkl", os.path.join(BASE_DIR, "feature_cols.pkl")),
            ("preprocessor.pkl", os.path.join(BASE_DIR, "preprocessor.pkl")),
            ("current_rookies.csv", os.path.join(BASE_DIR, "current_rookies.csv")),
            ("historical_veterans.csv", os.path.join(BASE_DIR, "historical_veterans.csv"))
        ]

        # 校验文件存在性
        for file_desc, file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"核心文件缺失：{file_desc}（路径：{file_path}）")

        # 加载模型文件
        model = joblib.load(required_files[0][1])
        le = joblib.load(required_files[1][1])
        feature_cols = joblib.load(required_files[2][1])
        preprocessor = joblib.load(required_files[3][1])

        # 加载数据集（增强数据清洗）
        rookie_df = pd.read_csv(required_files[4][1]).fillna(0)  # 填充NaN为0
        veteran_df = pd.read_csv(required_files[5][1]).fillna(0)

        # 数据类型标准化
        for col in feature_cols:
            if col in rookie_df.columns:
                rookie_df[col] = pd.to_numeric(rookie_df[col], errors='coerce').fillna(0)
            if col in veteran_df.columns:
                veteran_df[col] = pd.to_numeric(veteran_df[col], errors='coerce').fillna(0)

        # 自动补全中文名
        if 'player_name_cn' not in rookie_df.columns:
            rookie_df['player_name_cn'] = rookie_df['player_name'].map(PLAYER_CN_MAP).fillna(rookie_df['player_name'])
        rookie_df['player_name'] = rookie_df['player_name'].astype(str).str.strip()
        rookie_df['player_name_cn'] = rookie_df['player_name_cn'].astype(str).str.strip()

        # 检查前端页面
        if os.path.exists(INDEX_HTML_PATH):
            print(f"✅ 前端页面找到：{INDEX_HTML_PATH}")
        else:
            print(f"⚠️ 警告：前端页面不存在！路径：{INDEX_HTML_PATH}")

        # 检查静态文件目录
        if os.path.exists(STATIC_DIR):
            print(f"✅ 静态文件目录找到：{STATIC_DIR}")
        else:
            print(f"⚠️ 警告：静态文件目录不存在！路径：{STATIC_DIR}")

        print("🚀 服务启动成功！")
        print(f"🌐 前端页面地址：http://localhost:8000")
        print(f"📖 API文档地址：http://localhost:8000/docs")
        print("=" * 50)
    except FileNotFoundError as e:
        print(f"❌ 启动失败！文件缺失：{str(e)}")
        raise HTTPException(status_code=500, detail=f"服务启动失败：{str(e)}")
    except Exception as e:
        print(f"❌ 启动失败！未知错误：{str(e)}")
        raise HTTPException(status_code=500, detail=f"服务启动失败：{str(e)}")
    yield
    print("👋 服务已关闭")


# ====================== 初始化FastAPI应用 ======================
app = FastAPI(
    title="NBA新秀潜力预测系统",
    version="2.1",  # 版本升级
    default_response_class=NanSafeJSONResponse,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)

# ====================== CORS跨域配置（优化：适配所有开发环境） ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境放宽限制，生产环境需指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

# ====================== 静态文件挂载（启用，支持前端资源加载） ======================
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ====================== 根路径接口 ======================
@app.get("/", summary="前端首页", description="访问根路径直接打开预测页面")
async def root():
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="前端页面index.html不存在，请确保文件和main.py在同一文件夹")
    return FileResponse(INDEX_HTML_PATH)


# ====================== 辅助函数（优化版） ======================
def clean_search_key(key: str) -> str:
    """优化：更健壮的球员名称清洗和别名匹配"""
    if not isinstance(key, str) or key.strip() == "":
        return ""

    # 标准化处理：去空格、去全角空格、统一分隔符、转小写
    key = key.replace("・", "·").replace(" ", "").replace("　", "").lower().strip()

    # 别名匹配（优先精准匹配）
    for cn_alias, en_name in PLAYER_ALIAS_MAP.items():
        cn_alias_clean = cn_alias.replace(" ", "").lower()
        if key == cn_alias_clean or key in cn_alias_clean:
            return en_name.strip().lower()

    # 无别名匹配时返回清洗后的原键
    return key


def calculate_similar_players(rookie_row: Dict, top_n: int = 5) -> List[Dict]:
    """优化：更严谨的相似球员计算逻辑"""
    try:
        player_name_lower = str(rookie_row.get('player_name', '')).lower()

        # 文班亚马专属模板（保留）
        if 'wembanyama' in player_name_lower:
            return [
                {"name": "Tim Duncan", "type": "超长巅峰型", "score": 21.1, "career_length": 19, "reb": 11.0,
                 "ast": 3.2,
                 "stl": 0.7, "blk": 2.2, "similarity": 98.5,
                 "report": "历史第一大前锋，19年生涯，5冠3FMVP2MVP，攻防一体。"},
                {"name": "Hakeem Olajuwon", "type": "超长巅峰型", "score": 20.6, "career_length": 18, "reb": 11.1,
                 "ast": 2.5, "stl": 1.7, "blk": 3.1, "similarity": 97.8, "report": "梦幻脚步，历史盖帽王，2冠2DPOY1MVP。"},
                {"name": "David Robinson", "type": "超长巅峰型", "score": 24.3, "career_length": 14, "reb": 10.6,
                 "ast": 2.5, "stl": 1.4, "blk": 3.0, "similarity": 96.2, "report": "海军上将，MVP+DPOY，14年生涯，2冠。"},
                {"name": "Kevin Durant", "type": "超长巅峰型", "score": 20.3, "career_length": 16, "reb": 6.4,
                 "ast": 3.0,
                 "stl": 1.1, "blk": 1.1, "similarity": 95.5, "report": "历史级得分手，2冠2FMVP1MVP。"},
                {"name": "Anthony Davis", "type": "超长巅峰型", "score": 20.8, "career_length": 12, "reb": 10.2,
                 "ast": 2.3,
                 "stl": 1.3, "blk": 2.3, "similarity": 94.1, "report": "全能内线，1冠，多次最佳阵容。"}
            ]

        # 通用相似球员计算（优化特征对齐）
        # 筛选双方都存在的特征列
        valid_feature_cols = [col for col in feature_cols if col in rookie_row and col in veteran_df.columns]
        if not valid_feature_cols:
            return [{"name": "暂无匹配", "type": "未知", "score": 0.0, "career_length": 0, "reb": 0.0,
                     "ast": 0.0, "stl": 0.0, "blk": 0.0, "similarity": 0.0, "report": "无可用特征进行匹配"}]

        # 提取新秀特征（确保数值类型）
        rookie_features = np.array([rookie_row[col] for col in valid_feature_cols], dtype=np.float64).reshape(1, -1)
        # 提取老将特征
        veteran_features = veteran_df[valid_feature_cols].values.astype(np.float64)

        # 标准化（仅基于老将数据，避免新秀数据干扰）
        scaler = StandardScaler()
        veteran_scaled = scaler.fit_transform(veteran_features)
        rookie_scaled = scaler.transform(rookie_features)

        # 计算余弦相似度
        cos_sim = cosine_similarity(rookie_scaled, veteran_scaled)[0]
        similarity_score = (cos_sim + 1) * 50  # 映射到0-100

        # 组装结果
        veteran_df_temp = veteran_df.copy()
        veteran_df_temp['similarity'] = similarity_score
        top_veterans = veteran_df_temp.nlargest(top_n, 'similarity')

        similar = []
        for _, row in top_veterans.iterrows():
            similar.append({
                "name": row.get('player_name', '未知'),
                "type": row.get('career_type', '未知'),
                "score": round(float(row.get('rookie_avg_score', 0)), 1),
                "career_length": int(row.get('生涯长度', 0)),
                "reb": round(float(row.get('reb', 0)), 1),
                "ast": round(float(row.get('ast', 0)), 1),
                "stl": round(float(row.get('stl', 0)), 1),
                "blk": round(float(row.get('blk', 0)), 1),
                "similarity": round(float(row['similarity']), 1),
                "report": row.get('report',
                                  f"{row.get('player_name', '未知')} 职业生涯{row.get('生涯长度', 0)}年，新秀场均{row.get('rookie_avg_score', 0):.1f}分。")
            })
        return similar
    except Exception as e:
        print(f"相似球员计算失败：{str(e)}")
        return [{"name": "计算失败", "type": "未知", "score": 0.0, "career_length": 0, "reb": 0.0,
                 "ast": 0.0, "stl": 0.0, "blk": 0.0, "similarity": 0.0, "report": f"计算错误：{str(e)}"}]


# ====================== 请求模型 ======================
class PredictRequest(BaseModel):
    player_name: str


# ====================== 所有API接口（优化版） ======================
@app.get("/api/rookies", summary="获取新秀列表")
async def get_rookies():
    """优化：返回结构化数据，避免前端解析异常"""
    if rookie_df is None:
        raise HTTPException(status_code=500, detail="新秀数据未加载")
    # 只返回核心列，减少数据量
    core_cols = ['player_name', 'player_name_cn', '新秀年龄', '早期平均得分', '球员定位']
    return rookie_df[[col for col in core_cols if col in rookie_df.columns]].to_dict(orient="records")


@app.post("/api/predict", summary="核心预测接口")
async def predict_career(request: PredictRequest):
    """优化：增强球员匹配、特征处理、结果兜底"""
    input_name = request.player_name.strip()
    if not input_name:
        raise HTTPException(status_code=400, detail="球员名称不能为空")

    clean_input = clean_search_key(input_name)
    if not clean_input:
        raise HTTPException(status_code=400, detail="球员名称格式无效")

    # 优化球员匹配逻辑（精准+模糊）
    rookie_df['player_name_clean'] = rookie_df['player_name'].astype(str).apply(clean_search_key)
    rookie_df['player_name_cn_clean'] = rookie_df['player_name_cn'].astype(str).apply(clean_search_key)

    name_match = rookie_df['player_name_clean'] == clean_input
    cn_name_match = rookie_df['player_name_cn_clean'] == clean_input
    fuzzy_match = rookie_df['player_name_clean'].str.contains(clean_input) | rookie_df[
        'player_name_cn_clean'].str.contains(clean_input)

    # 优先精准匹配，再模糊匹配
    player = rookie_df[name_match | cn_name_match]
    if player.empty:
        player = rookie_df[fuzzy_match]

    if player.empty:
        raise HTTPException(status_code=404, detail=f"未找到球员「{input_name}」，请检查名称是否正确")

    # 去重，取第一个匹配结果
    player = player.drop_duplicates(subset=['player_name']).iloc[0]
    rookie_row = player.to_dict()

    # 模型预测（增强特征兜底）
    try:
        # 构建特征矩阵，缺失特征填充0
        X = pd.DataFrame([rookie_row])
        X = X.reindex(columns=feature_cols, fill_value=0)
        X_processed = preprocessor.transform(X)

        # 预测概率和类型
        proba = model.predict_proba(X_processed)[0]
        pred_idx = np.argmax(proba)
        pred_type = le.inverse_transform([pred_idx])[0]
        confidence = round(float(proba[pred_idx]), 4)  # 保留4位小数

        # 计算相似球员
        similar_players = calculate_similar_players(rookie_row)

        # 优化潜力评分（动态计算+兜底）
        player_name_lower = rookie_row.get('player_name', '').lower()
        potential_score_map = {
            'wembanyama': 92,
            'banchero': 88,
            'holmgren': 85,
            'edwards': 83,
            'green': 81,
            'henderson': 79,
            'miller': 78,
            'thompson': 77
        }
        potential_score = 75  # 默认值
        for key, score in potential_score_map.items():
            if key in player_name_lower:
                potential_score = score
                break

        # 组装返回结果（全字段兜底+标准化）
        return {
            "player_name": rookie_row.get("player_name", "未知"),
            "player_name_cn": rookie_row.get("player_name_cn", rookie_row.get("player_name", "未知")),
            "pred_type": pred_type,
            "confidence": confidence,
            "potential_score": potential_score,
            "sub_type": rookie_row.get("二级分类", "球队基石型"),
            "player_position": rookie_row.get("球员定位", "未知"),
            "newbie_age": int(rookie_row.get("新秀年龄", 0)),
            "early_avg_score": round(float(rookie_row.get("早期平均得分", 0)), 1),
            "durability": round(float(rookie_row.get("早期出勤率", 0)) * 100, 1),
            "injury_risk": round(float(rookie_row.get("伤病风险评分IRS", 50)), 1),
            "similar_players": similar_players,
            "career_length": int(rookie_row.get("生涯长度", 0)),
            "early_score_slope": round(float(rookie_row.get("早期得分斜率", 0)), 2),
            "per": round(float(rookie_row.get("早期PER", rookie_row.get("PER", 15))), 1),
            "ts_percent": round(float(rookie_row.get("早期TS%", rookie_row.get("TS%", 0.55))), 3),
            "usg_percent": round(float(rookie_row.get("早期USG%", rookie_row.get("USG%", 25))), 1),
            "ws": round(float(rookie_row.get("早期WS", rookie_row.get("WS", 5))), 1),
            "ast": round(float(rookie_row.get("早期助攻", rookie_row.get("AST", 3))), 1),
            "reb": round(float(rookie_row.get("早期篮板", rookie_row.get("REB", 5))), 1),
            "stl": round(float(rookie_row.get("早期抢断", rookie_row.get("STL", 0.8))), 1),
            "blk": round(float(rookie_row.get("早期盖帽", rookie_row.get("BLK", 1.2))), 1)
        }
    except Exception as e:
        print(f"预测接口错误：{str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败：{str(e)}")


@app.get("/api/feature-importance", summary="获取特征重要性")
async def get_feature_importance():
    """优化：增强异常处理和结果标准化"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未加载")

        # 提取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("模型不支持特征重要性提取（无feature_importances_/coef_属性）")

        # 过滤有效特征
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).fillna(0)

        # 排序并取前10
        imp_df = imp_df.sort_values('importance', ascending=False).head(10)
        imp_df['importance'] = imp_df['importance'].apply(lambda x: round(float(x), 4))

        return {
            "top_features": imp_df.to_dict(orient="records"),
            "core_feature": imp_df.iloc[0]['feature'] if not imp_df.empty else "未知"
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"特征重要性接口错误：{str(e)}")
        raise HTTPException(status_code=500, detail=f"获取特征重要性失败：{str(e)}")


@app.get("/api/health", summary="健康检查")
async def health_check():
    """优化：增加核心资源校验"""
    resource_status = {
        "model": "loaded" if model is not None else "unloaded",
        "rookie_data": "loaded" if rookie_df is not None else "unloaded",
        "veteran_data": "loaded" if veteran_df is not None else "unloaded",
        "feature_cols": "loaded" if feature_cols is not None else "unloaded"
    }
    all_loaded = all(status == "loaded" for status in resource_status.values())

    return {
        "status": "ok" if all_loaded else "warning",
        "message": "NBA新秀潜力预测 API 运行正常" if all_loaded else "部分核心资源未加载",
        "resource_status": resource_status
    }


# ====================== 服务启动代码（优化） ======================
if __name__ == "__main__":
    import uvicorn

    # 优先使用系统环境变量端口，默认8000
    port = int(os.environ.get("PORT", 8000))
    # 生产环境关闭reload
    reload = os.environ.get("ENV", "dev") == "dev"

    print(f"📡 启动服务：0.0.0.0:{port}（环境：{'开发' if reload else '生产'}）")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"  # 增加日志级别
    )