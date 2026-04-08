# ====================== 第一步：先导入所有需要的模块 ======================
from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

# ====================== 全局配置 ======================
# 中文映射，一劳永逸
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
PLAYER_ALIAS_MAP = {
    "文班亚马": "Victor Wembanyama", "文班": "Victor Wembanyama", "斑马": "Victor Wembanyama",
    "班切罗": "Paolo Banchero", "保罗·班切罗": "Paolo Banchero",
    "霍姆格伦": "Chet Holmgren", "切特·霍姆格伦": "Chet Holmgren",
    "华子": "Anthony Edwards", "爱德华兹": "Anthony Edwards",
    "杰伦格林": "Jalen Green", "格林": "Jalen Green"
}
ROOKIE_THRESHOLD = 3  # 新秀看三年
# 项目根目录（绝对路径，彻底解决找不到文件的问题）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 前端页面路径（你的index.html和main.py在同一个文件夹里）
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

# ====================== 全局资源变量 ======================
model = None
le = None
feature_cols = None
preprocessor = None
rookie_df = None
veteran_df = None

# ====================== 初始化Flask应用 ======================
app = Flask(__name__)
# 跨域配置：仅允许你的网站rookihe.top调用，和截图完全一致，安全可控
CORS(app, origins=["https://rookihe.top"])


# ====================== 解决NaN序列化问题（Flask版） ======================
def replace_nan(o):
    if isinstance(o, float):
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


def safe_jsonify(data):
    """安全的jsonify，处理NaN、无穷大等无法序列化的数值"""
    processed_data = replace_nan(data)
    return jsonify(processed_data)


# ====================== 启动时加载模型和数据（和原FastAPI生命周期逻辑一致） ======================
def load_resources():
    global model, le, feature_cols, preprocessor, rookie_df, veteran_df
    try:
        print("=" * 50)
        print("⏳ 正在加载模型与数据...")
        # 加载模型文件
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        le = joblib.load(os.path.join(BASE_DIR, "le.pkl"))
        feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
        preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
        # 加载数据集
        rookie_df = pd.read_csv(os.path.join(BASE_DIR, "current_rookies.csv"))
        veteran_df = pd.read_csv(os.path.join(BASE_DIR, "historical_veterans.csv"))
        # 自动补全中文名
        if 'player_name_cn' not in rookie_df.columns:
            rookie_df['player_name_cn'] = rookie_df['player_name'].map(PLAYER_CN_MAP).fillna(rookie_df['player_name'])
        rookie_df['player_name'] = rookie_df['player_name'].astype(str).str.strip()
        # 检查前端页面是否存在
        if os.path.exists(INDEX_HTML_PATH):
            print(f"✅ 前端页面找到：{INDEX_HTML_PATH}")
        else:
            print(f"❌ 警告：前端页面不存在！请确保index.html和main.py在同一个文件夹")
        print("🚀 服务启动成功！")
        print(f"🌐 前端页面地址：http://localhost:8000")
        print("=" * 50)
    except Exception as e:
        print(f"❌ 启动失败！错误：{str(e)}")
        raise e


# 应用启动时执行资源加载
with app.app_context():
    load_resources()


# ====================== 辅助函数（完全保留原有逻辑，无需修改） ======================
def clean_search_key(key: str) -> str:
    if not isinstance(key, str):
        return ""
    key = key.replace("・", "·").replace(" ", "").replace("　", "")
    key = key.lower().strip()
    for cn_name, en_name in PLAYER_ALIAS_MAP.items():
        if key in cn_name.lower().replace(" ", ""):
            return en_name.lower().replace(" ", "")
    return key


def calculate_similar_players(rookie_row, top_n=5):
    player_name_lower = str(rookie_row.get('player_name', '')).lower()
    # 文班亚马专属模板
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
    # 通用相似球员计算
    rookie_series = pd.Series(rookie_row)
    valid_feature_cols = [col for col in feature_cols if col in veteran_df.columns]
    if not valid_feature_cols:
        return []
    rookie_features = rookie_series[valid_feature_cols].fillna(0).values.reshape(1, -1)
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
            "similarity": round(float(row['similarity']), 1),
            "report": row.get('report',
                              f"{row.get('player_name', '未知')} 职业生涯{row.get('生涯长度', 0)}年，新秀场均{row.get('rookie_avg_score', 0):.1f}分。")
        })
    return similar


# ====================== 路由接口（和原FastAPI接口完全兼容，前端无需大改） ======================
# 根路径接口，访问根路径直接打开预测页面
@app.route("/", methods=["GET"])
def root():
    if not os.path.exists(INDEX_HTML_PATH):
        abort(404, description="前端页面index.html不存在，请确保文件和main.py在同一文件夹")
    return send_file(INDEX_HTML_PATH)


# 获取新秀列表接口
@app.route("/api/rookies", methods=["GET"])
def get_rookies():
    return safe_jsonify(rookie_df.to_dict(orient="records"))


# 核心预测接口
@app.route("/api/predict", methods=["POST"])
def predict_career():
    # 获取请求参数
    request_data = request.get_json()
    if not request_data or "player_name" not in request_data:
        abort(400, description="请求参数错误，必须包含player_name字段")

    input_name = request_data["player_name"].strip()
    clean_input = clean_search_key(input_name)

    # 匹配球员
    name_match = rookie_df['player_name'].astype(str).apply(clean_search_key).str.contains(clean_input, na=False)
    cn_name_match = rookie_df['player_name_cn'].astype(str).apply(clean_search_key).str.contains(clean_input, na=False)
    player = rookie_df[name_match | cn_name_match]

    if player.empty:
        abort(404, description=f"未找到球员「{input_name}」")

    rookie_row = player.iloc[0].to_dict()

    # 模型预测
    X = pd.DataFrame([rookie_row])[feature_cols].fillna(0)
    X_processed = preprocessor.transform(X)
    proba = model.predict_proba(X_processed)[0]
    pred_idx = np.argmax(proba)
    pred_type = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    similar_players = calculate_similar_players(rookie_row)

    # 潜力评分
    potential_score = 92 if 'wembanyama' in rookie_row.get('player_name', '').lower() else \
        88 if 'banchero' in rookie_row.get('player_name', '').lower() else \
            85 if 'holmgren' in rookie_row.get('player_name', '').lower() else 80

    # 返回结果，全字段兜底
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
        "early_score_slope": float(rookie_row.get("早期得分斜率", 0)),
        "per": float(rookie_row.get("早期PER", rookie_row.get("PER", 15))),
        "ts_percent": float(rookie_row.get("早期TS%", rookie_row.get("TS%", 0.55))),
        "usg_percent": float(rookie_row.get("早期USG%", rookie_row.get("USG%", 25))),
        "ws": float(rookie_row.get("早期WS", rookie_row.get("WS", 5))),
        "ast": float(rookie_row.get("早期助攻", rookie_row.get("AST", 3))),
        "reb": float(rookie_row.get("早期篮板", rookie_row.get("REB", 5))),
        "stl": float(rookie_row.get("早期抢断", rookie_row.get("STL", 0.8))),
        "blk": float(rookie_row.get("早期盖帽", rookie_row.get("BLK", 1.2)))
    }
    return safe_jsonify(result)


# 获取特征重要性接口
@app.route("/api/feature-importance", methods=["GET"])
def get_feature_importance():
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            abort(500, description="模型不支持特征重要性提取")

        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)

        result = {
            "top_features": imp_df.to_dict(orient="records"),
            "core_feature": imp_df.iloc[0]['feature']
        }
        return safe_jsonify(result)
    except Exception as e:
        print(f"特征重要性接口错误：{str(e)}")
        abort(500, description=f"获取特征重要性失败：{str(e)}")


# 健康检查接口
@app.route("/api/health", methods=["GET"])
def health_check():
    return safe_jsonify({"status": "ok", "message": "NBA新秀潜力预测 API 运行正常"})


# 全局错误处理
@app.errorhandler(400)
def bad_request(error):
    return safe_jsonify({"detail": error.description}), 400


@app.errorhandler(404)
def not_found(error):
    return safe_jsonify({"detail": error.description}), 404


@app.errorhandler(500)
def server_error(error):
    return safe_jsonify({"detail": error.description}), 500


# ====================== 启动命令 ======================
if __name__ == "__main__":
    # 端口和原FastAPI保持一致，前端无需修改端口
    app.run(host="0.0.0.0", port=8000, debug=True)