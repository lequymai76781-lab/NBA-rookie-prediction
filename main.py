from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

import os
import re
import json
import math
import joblib
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ====================== 全局配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
STATIC_DIR = os.path.join(BASE_DIR, "static")
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime_assets")
if not os.path.isdir(RUNTIME_DIR):
    print(f"警告：未找到 runtime_assets，回退到项目根目录: {BASE_DIR}")
    RUNTIME_DIR = BASE_DIR
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

PLAYER_CN_MAP = {
    "Victor Wembanyama": "维克托·文班亚马",
    "Paolo Banchero": "保罗·班切罗",
    "Chet Holmgren": "切特·霍姆格伦",
    "Anthony Edwards": "安东尼·爱德华兹",
    "Jalen Green": "杰伦·格林",
    "Scoot Henderson": "斯库特·亨德森",
    "Brandon Miller": "布兰登·米勒",
    "Amen Thompson": "阿门·汤普森",
    "Ausar Thompson": "奥萨尔·汤普森",
}

PLAYER_ALIAS_MAP = {
    "文班亚马": "Victor Wembanyama",
    "文班": "Victor Wembanyama",
    "斑马": "Victor Wembanyama",
    "班切罗": "Paolo Banchero",
    "保罗班切罗": "Paolo Banchero",
    "霍姆格伦": "Chet Holmgren",
    "切特霍姆格伦": "Chet Holmgren",
    "华子": "Anthony Edwards",
    "爱德华兹": "Anthony Edwards",
    "杰伦格林": "Jalen Green",
    "格林": "Jalen Green",
    "亨德森": "Scoot Henderson",
    "米勒": "Brandon Miller",
    "阿门汤普森": "Amen Thompson",
    "奥萨尔汤普森": "Ausar Thompson",
}

stage1_special_thresholds = {1: 0.28, 2: 0.26, 3: 0.24}

# ====================== 全局资源 ======================
feature_cols = None

primary_stage1_models = None
primary_stage1_imputers = None
primary_stage1_scalers = None
primary_stage1_label_encoders = None

primary_special_models = None
primary_special_imputers = None
primary_special_scalers = None
primary_special_label_encoders = None

rookie_df = None
veteran_df = None
metrics_json = {}


# ====================== JSON 安全响应 ======================
class NanSafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        def replace_nan(o):
            if isinstance(o, float):
                return 0.0 if not math.isfinite(o) else o
            if isinstance(o, np.floating):
                return 0.0 if not np.isfinite(o) else float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, dict):
                return {k: replace_nan(v) for k, v in o.items()}
            if isinstance(o, list):
                return [replace_nan(v) for v in o]
            return o

        return json.dumps(
            replace_nan(content),
            ensure_ascii=False,
            separators=(",", ":")
        ).encode("utf-8")


# ====================== 请求模型 ======================
class PredictRequest(BaseModel):
    player_name: str
    window_years: Optional[int] = None


class CompareRequest(BaseModel):
    player_name_a: str
    player_name_b: str


# ====================== 工具函数 ======================
def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件: {path}")
    return joblib.load(path)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = s.replace("・", "·").replace("　", "").replace(" ", "")
    s = re.sub(r"\s+", "", s)
    return s


def clean_search_key(key: str) -> str:
    key = normalize_name(key)
    if not key:
        return ""
    for alias, canonical in PLAYER_ALIAS_MAP.items():
        if key == normalize_name(alias):
            return normalize_name(canonical)
    return key


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def infer_player_cn(name: str) -> str:
    return PLAYER_CN_MAP.get(name, name)


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def preprocess_for_model(row_df: pd.DataFrame, imputer, scaler) -> np.ndarray:
    row_df = row_df.copy()
    for c in row_df.columns:
        if row_df[c].isna().all():
            row_df[c] = 0.0
    arr = imputer.transform(row_df)
    arr = scaler.transform(arr)
    return arr


def select_available_window(window_year: int, model_dict: Dict[int, Any]) -> int:
    if window_year in model_dict:
        return int(window_year)
    available = sorted(model_dict.keys())
    if not available:
        raise ValueError("没有可用模型")
    return min(available, key=lambda x: abs(x - window_year))


def primary_base_score(label: str) -> int:
    mapping = {
        "超长巅峰型": 94,
        "长青稳定型": 82,
        "大器晚成型": 79,
        "生涯波动型": 74,
        "早衰型": 66,
        "特殊轨迹型": 76,
    }
    return mapping.get(label, 75)


def primary_career_length_text(label: str) -> str:
    mapping = {
        "超长巅峰型": "12-18",
        "长青稳定型": "9-14",
        "大器晚成型": "8-13",
        "生涯波动型": "6-11",
        "早衰型": "4-8",
    }
    return mapping.get(label, "8-12")


def primary_peak_period_text(label: str) -> str:
    mapping = {
        "超长巅峰型": "2-4",
        "长青稳定型": "3-5",
        "大器晚成型": "4-6",
        "生涯波动型": "2-5",
        "早衰型": "1-3",
    }
    return mapping.get(label, "3-5")


def stage_meta(window_years: int, final_label: str) -> Dict[str, Any]:
    if window_years == 1:
        return {
            "analysis_stage": "早期观察",
            "confidence_level": "低",
            "primary_result_type": "一级方向参考",
            "primary_result_note": "当前仅基于首年数据，结果用于早期观察与风险预警，不作为正式生涯轨迹结论。",
            "official_primary_label": None,
        }
    if window_years == 2:
        return {
            "analysis_stage": "中期判断",
            "confidence_level": "中",
            "primary_result_type": "一级趋势判断",
            "primary_result_note": "当前基于前两年数据，结果用于中期趋势判断，正式结论以三年窗口结果为主。",
            "official_primary_label": None,
        }
    return {
        "analysis_stage": "正式预测",
        "confidence_level": "高",
        "primary_result_type": "一级正式预测",
        "primary_result_note": "当前已达到三年窗口，结果作为系统正式主预测输出。",
        "official_primary_label": final_label,
    }


def build_core_strength_weakness(row: Dict[str, Any]) -> Tuple[str, str]:
    strengths, weaknesses = [], []
    if safe_float(row.get("points_per36")) >= 20:
        strengths.append("高产得分")
    if safe_float(row.get("assists_per36")) >= 6:
        strengths.append("组织发起")
    if safe_float(row.get("rebounds_per36")) >= 9:
        strengths.append("篮板控制")
    if safe_float(row.get("blocks_per36")) >= 1.5:
        strengths.append("护框协防")
    if safe_float(row.get("three_point_pct")) >= 0.36:
        strengths.append("外线投射")
    if safe_float(row.get("true_shooting_pct")) >= 0.58:
        strengths.append("终结效率")
    if safe_float(row.get("availability_mean")) >= 0.88:
        strengths.append("出勤稳定")

    if safe_float(row.get("three_point_pct")) <= 0.31:
        weaknesses.append("外线稳定性")
    if safe_float(row.get("true_shooting_pct")) <= 0.52:
        weaknesses.append("终结效率")
    if safe_float(row.get("ast_to_ratio")) <= 1.6:
        weaknesses.append("决策控制")
    if safe_float(row.get("availability_mean")) <= 0.72:
        weaknesses.append("出勤风险")
    if safe_float(row.get("minutes_cv")) >= 0.35:
        weaknesses.append("角色波动")

    strength_text = "、".join(strengths[:3]) if strengths else "身体天赋、成长空间"
    weakness_text = "、".join(weaknesses[:3]) if weaknesses else "比赛经验、稳定性"
    return strength_text, weakness_text


def build_rule_secondary_label(row: Dict[str, Any]) -> str:
    pts36 = safe_float(row.get("points_per36"), np.nan)
    ast36 = safe_float(row.get("assists_per36"), np.nan)
    reb36 = safe_float(row.get("rebounds_per36"), np.nan)
    blk36 = safe_float(row.get("blocks_per36"), np.nan)
    stl36 = safe_float(row.get("steals_per36"), np.nan)
    ts = safe_float(row.get("true_shooting_pct"), np.nan)
    usg = safe_float(row.get("usage_pct"), np.nan)
    tp = safe_float(row.get("three_point_pct"), np.nan)
    dws = safe_float(row.get("defensive_win_shares_cum"), np.nan)
    atr = safe_float(row.get("ast_to_ratio"), np.nan)
    per_mean = safe_float(row.get("per_mean"), np.nan)
    availability_mean = safe_float(row.get("availability_mean"), np.nan)
    minutes_trend = safe_float(row.get("minutes_trend"), np.nan)
    minutes_cv = safe_float(row.get("minutes_cv"), np.nan)
    window_years = safe_int(row.get("window_years"), 1)

    if pts36 >= 20 and ts >= 0.55 and usg >= 0.25:
        return "高效得分核心"
    if ast36 >= 7 and atr >= 2.5 and pts36 >= 15:
        return "组织核心型"
    if tp >= 0.37 and (stl36 >= 1.2 or dws >= 1.0) and 10 <= pts36 <= 18:
        return "3D侧翼型"
    if blk36 >= 1.8 and reb36 >= 10 and pts36 <= 16:
        return "护框内线型"
    if tp >= 0.35 and blk36 >= 1.0 and reb36 >= 7:
        return "空间内线型"
    if pts36 >= 18 and ts < 0.52 and usg >= 0.26:
        return "低效得分手型"
    if (stl36 >= 1.5 or blk36 >= 1.2) and dws >= 1.5 and pts36 <= 12:
        return "防守尖兵型"
    if window_years >= 3 and availability_mean <= 0.60 and minutes_cv >= 0.35:
        return "伤病风险型"
    if window_years >= 3 and minutes_trend > 0 and pts36 <= 12 and per_mean <= 14:
        return "潜力待挖掘型"
    return "潜力待定型"


def build_secondary_rule_explanation(row: Dict[str, Any], secondary_label: str) -> Dict[str, Any]:
    return {
        "type": "rule_based",
        "predicted_class": secondary_label,
        "core_metrics": {
            "points_per36": safe_float(row.get("points_per36")),
            "assists_per36": safe_float(row.get("assists_per36")),
            "rebounds_per36": safe_float(row.get("rebounds_per36")),
            "blocks_per36": safe_float(row.get("blocks_per36")),
            "steals_per36": safe_float(row.get("steals_per36")),
            "true_shooting_pct": safe_float(row.get("true_shooting_pct")),
            "usage_pct": safe_float(row.get("usage_pct")),
            "three_point_pct": safe_float(row.get("three_point_pct")),
            "per_mean": safe_float(row.get("per_mean")),
            "availability_mean": safe_float(row.get("availability_mean")),
        },
        "note": "二级分类由规则系统直接生成，不使用监督模型。",
    }


def build_radar_data(row: Dict[str, Any]) -> List[float]:
    scoring = min(100, max(0, safe_float(row.get("points_per36")) * 4))
    playmaking = min(100, max(0, safe_float(row.get("assists_per36")) * 10))
    rebounding = min(100, max(0, safe_float(row.get("rebounds_per36")) * 8))
    defense = min(100, max(0, (safe_float(row.get("steals_per36")) * 25 + safe_float(row.get("blocks_per36")) * 20)))
    shooting = min(100, max(0, safe_float(row.get("three_point_pct")) * 220))
    efficiency = min(100, max(0, safe_float(row.get("true_shooting_pct")) * 170))
    return [round(v, 1) for v in [defense, scoring, rebounding, playmaking, shooting, efficiency]]


def find_plot_url(player_name: str, rookie_start_season: int, window_years: int, suffix: str) -> Optional[str]:
    safe_name = str(player_name).replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}_{rookie_start_season}_y{window_years}_{suffix}.png"
    file_path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(file_path):
        return f"/plots/{filename}"
    return None


def build_primary_report(primary_label: str, secondary_label: str, analysis_stage: str) -> str:
    reports = {
        "超长巅峰型": "该球员具备长期成为球队核心的潜质，若健康与战术环境稳定，未来上限接近长期基石级球员。",
        "长青稳定型": "该球员更可能成长为长期稳定输出型主力，生涯曲线不会极端爆发，但持续性较强。",
        "大器晚成型": "该球员当前可能尚未完全兑现天赋，但成长斜率具备后期上扬空间，适合耐心培养。",
        "生涯波动型": "该球员未来可能出现明显阶段性波动，环境、角色与健康对其生涯走势影响较大。",
        "早衰型": "该球员早期可能迅速打出表现，但后期持续性存在隐患，需重点关注健康与负荷管理。",
        "特殊轨迹型": "该球员可能走向非常规发展轨迹，短期内需持续追踪成长趋势。",
    }
    stage_prefix = {
        "早期观察": "当前仍处于首年观察阶段，",
        "中期判断": "当前基于前两年表现进行趋势识别，",
        "正式预测": "当前已经形成三年窗口正式判断，",
    }
    base = reports.get(primary_label, "该球员具备一定成长空间，但发展路径仍需更多样本观察。")
    return f"{stage_prefix.get(analysis_stage, '')}{base} 当前规则画像倾向于「{secondary_label}」。"


def find_player_rows_by_query(query: str) -> pd.DataFrame:
    if rookie_df is None or rookie_df.empty:
        return pd.DataFrame()

    clean_q = clean_search_key(query)
    if not clean_q:
        return pd.DataFrame()

    df = rookie_df.copy()
    exact_mask = (df["player_name_norm"] == clean_q) | (df["player_name_cn_norm"] == clean_q)
    if exact_mask.any():
        return df[exact_mask].copy()

    fuzzy_mask = df["player_name_norm"].str.contains(clean_q, na=False) | df["player_name_cn_norm"].str.contains(clean_q, na=False)
    return df[fuzzy_mask].copy()


def predict_primary_two_stage(row: pd.Series) -> Dict[str, Any]:
    window_years = safe_int(row.get("window_years"), 1)
    stage1_window = select_available_window(window_years, primary_stage1_models)

    row_df = pd.DataFrame([row[feature_cols].to_dict()])[feature_cols]
    X_stage1 = preprocess_for_model(row_df, primary_stage1_imputers[stage1_window], primary_stage1_scalers[stage1_window])

    stage1_model = primary_stage1_models[stage1_window]
    stage1_le = primary_stage1_label_encoders[stage1_window]

    stage1_proba = stage1_model.predict_proba(X_stage1)[0]
    stage1_pred_idx = int(np.argmax(stage1_proba))
    stage1_classes = list(stage1_le.classes_)
    stage1_label = stage1_le.inverse_transform([stage1_pred_idx])[0]

    if "特殊轨迹型" in stage1_classes:
        special_idx = stage1_classes.index("特殊轨迹型")
        threshold = stage1_special_thresholds.get(stage1_window, 0.25)
        if stage1_proba[special_idx] >= threshold:
            stage1_pred_idx = special_idx
            stage1_label = "特殊轨迹型"

    stage1_conf = float(stage1_proba[stage1_pred_idx])
    final_label = stage1_label
    final_conf = stage1_conf
    special_window = None
    special_conf = None

    if stage1_label == "特殊轨迹型" and primary_special_models:
        special_window = select_available_window(window_years, primary_special_models)
        X_special = preprocess_for_model(row_df, primary_special_imputers[special_window], primary_special_scalers[special_window])
        special_model = primary_special_models[special_window]
        special_le = primary_special_label_encoders[special_window]

        special_proba = special_model.predict_proba(X_special)[0]
        special_pred_idx = int(np.argmax(special_proba))
        final_label = special_le.inverse_transform([special_pred_idx])[0]
        special_conf = float(special_proba[special_pred_idx])
        final_conf = special_conf

    return {
        "pred_primary_stage1_label": stage1_label,
        "pred_primary_stage1_confidence": round(stage1_conf, 4),
        "pred_primary_stage1_window_model": stage1_window,
        "pred_primary_special_window_model": special_window,
        "pred_primary_special_confidence": round(special_conf, 4) if special_conf is not None else None,
        "pred_primary_label": final_label,
        "pred_primary_confidence": round(final_conf, 4),
    }


def build_similar_players(row: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
    if veteran_df is None or veteran_df.empty:
        return []

    base_df = veteran_df.copy()
    if "window_years" in base_df.columns:
        same_window = base_df[base_df["window_years"] == row["window_years"]].copy()
        if not same_window.empty:
            base_df = same_window

    usable_cols = [c for c in feature_cols if c in base_df.columns]
    if not usable_cols:
        return []

    base_df = ensure_columns(base_df, usable_cols).copy()
    for c in usable_cols:
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce").fillna(0)

    query = pd.DataFrame([row[usable_cols].to_dict()])[usable_cols].copy()
    for c in usable_cols:
        query[c] = pd.to_numeric(query[c], errors="coerce").fillna(0)

    scaler = StandardScaler()
    veteran_scaled = scaler.fit_transform(base_df[usable_cols].values.astype(np.float64))
    query_scaled = scaler.transform(query.values.astype(np.float64))

    sims = cosine_similarity(query_scaled, veteran_scaled)[0]
    temp = base_df.copy()
    temp["similarity"] = sims * 100
    top = temp.sort_values("similarity", ascending=False).head(top_n)

    result = []
    for _, r in top.iterrows():
        result.append({
            "name": r.get("player_name", "未知"),
            "type": r.get("primary_label", "未知"),
            "sub_type": r.get("secondary_label", "未知"),
            "score": round(safe_float(r.get("points_per36")), 1),
            "career_length": safe_int(r.get("career_length", 0)),
            "reb": round(safe_float(r.get("rebounds_per36")), 1),
            "ast": round(safe_float(r.get("assists_per36")), 1),
            "stl": round(safe_float(r.get("steals_per36")), 1),
            "blk": round(safe_float(r.get("blocks_per36")), 1),
            "similarity": round(safe_float(r.get("similarity")), 1),
            "report": f"{r.get('player_name', '未知')} 的历史模板与当前球员在新秀期特征上高度接近，一级类型为 {r.get('primary_label', '未知')}，二级画像为 {r.get('secondary_label', '未知')}。"
        })
    return result


def build_prediction_payload(row: pd.Series) -> Dict[str, Any]:
    row = row.copy()
    primary_out = predict_primary_two_stage(row)

    secondary_label = row.get("secondary_label", None)
    if not secondary_label or pd.isna(secondary_label):
        secondary_label = build_rule_secondary_label(row.to_dict())

    meta = stage_meta(safe_int(row.get("window_years"), 1), primary_out["pred_primary_label"])
    strength_text, weakness_text = build_core_strength_weakness(row.to_dict())

    potential_score = primary_base_score(primary_out["pred_primary_label"])
    potential_score = int(round(potential_score * 0.75 + primary_out["pred_primary_confidence"] * 100 * 0.25))
    potential_score = max(55, min(97, potential_score))

    player_name = str(row.get("player_name", "未知"))
    rookie_start_season = safe_int(row.get("rookie_start_season"), 0)
    window_years = safe_int(row.get("window_years"), 1)

    payload = {
        "player_name": player_name,
        "player_name_cn": row.get("player_name_cn", infer_player_cn(player_name)),
        "player_position": row.get("position_mode", row.get("player_position", "未知")),
        "window_years": window_years,
        **meta,
        **primary_out,
        "potential_score": potential_score,
        "career_length": primary_career_length_text(primary_out["pred_primary_label"]),
        "peak_period": primary_peak_period_text(primary_out["pred_primary_label"]),
        "pred_secondary_label": secondary_label,
        "secondary_result_type": "规则画像",
        "newbie_age": round(safe_float(row.get("age_mean")), 1),
        "early_avg_score": round(safe_float(row.get("points_per36")), 1),
        "durability": round(safe_float(row.get("availability_mean")) * 100, 1),
        "injury_risk": round(min(100, max(0, (1 - safe_float(row.get("availability_mean"), 0.8)) * 100 + safe_float(row.get("minutes_cv"), 0) * 40)), 1),
        "per": round(safe_float(row.get("per_mean")), 1),
        "ts_percent": round(safe_float(row.get("true_shooting_pct")) * 100, 1),
        "usg_percent": round(safe_float(row.get("usage_pct")) * 100, 1),
        "ws": round(safe_float(row.get("win_shares_cum")), 1),
        "ast": round(safe_float(row.get("assists_per36")), 1),
        "reb": round(safe_float(row.get("rebounds_per36")), 1),
        "stl": round(safe_float(row.get("steals_per36")), 1),
        "blk": round(safe_float(row.get("blocks_per36")), 1),
        "core_strength": strength_text,
        "core_weakness": weakness_text,
        "report": build_primary_report(primary_out["pred_primary_label"], secondary_label, meta["analysis_stage"]),
        "secondary_explanation": build_secondary_rule_explanation(row.to_dict(), secondary_label),
        "radar_data": build_radar_data(row.to_dict()),
        "similar_players": build_similar_players(row, top_n=5),
        "primary_shap_plot": None,
        "primary_stage1_shap_plot": None,
        "primary_special_shap_plot": None,
    }

    if window_years == 3:
        payload["primary_stage1_shap_plot"] = find_plot_url(player_name, rookie_start_season, window_years, "primary_stage1_shap")
        payload["primary_special_shap_plot"] = find_plot_url(player_name, rookie_start_season, window_years, "primary_special_shap")
        payload["primary_shap_plot"] = payload["primary_special_shap_plot"] or payload["primary_stage1_shap_plot"]

    return payload


# ====================== 生命周期 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_cols
    global primary_stage1_models, primary_stage1_imputers, primary_stage1_scalers, primary_stage1_label_encoders
    global primary_special_models, primary_special_imputers, primary_special_scalers, primary_special_label_encoders
    global rookie_df, veteran_df, metrics_json

    try:
        print("=" * 60)
        print("⏳ 正在加载运行资源...")

        if not os.path.isdir(RUNTIME_DIR):
            raise FileNotFoundError(f"未找到 runtime_assets 目录: {RUNTIME_DIR}")

        os.makedirs(PLOTS_DIR, exist_ok=True)

        feature_cols = load_pickle(os.path.join(RUNTIME_DIR, "feature_cols.pkl"))

        primary_stage1_models = load_pickle(os.path.join(RUNTIME_DIR, "primary_stage1_models_by_window.pkl"))
        primary_stage1_imputers = load_pickle(os.path.join(RUNTIME_DIR, "primary_stage1_imputers_by_window.pkl"))
        primary_stage1_scalers = load_pickle(os.path.join(RUNTIME_DIR, "primary_stage1_scalers_by_window.pkl"))
        primary_stage1_label_encoders = load_pickle(os.path.join(RUNTIME_DIR, "primary_stage1_label_encoders_by_window.pkl"))

        primary_special_models = load_pickle(os.path.join(RUNTIME_DIR, "primary_special_models_by_window.pkl"))
        primary_special_imputers = load_pickle(os.path.join(RUNTIME_DIR, "primary_special_imputers_by_window.pkl"))
        primary_special_scalers = load_pickle(os.path.join(RUNTIME_DIR, "primary_special_scalers_by_window.pkl"))
        primary_special_label_encoders = load_pickle(os.path.join(RUNTIME_DIR, "primary_special_label_encoders_by_window.pkl"))

        metrics_path = os.path.join(RUNTIME_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_json = json.load(f)

        rookie_df = pd.read_csv(os.path.join(RUNTIME_DIR, "current_rookie_samples.csv")).fillna(np.nan)
        veteran_df = pd.read_csv(os.path.join(RUNTIME_DIR, "historical_training_samples.csv")).fillna(np.nan)

        rookie_df = ensure_columns(rookie_df, feature_cols + ["player_name", "rookie_start_season", "window_years", "secondary_label", "position_mode"])
        veteran_df = ensure_columns(veteran_df, feature_cols + ["player_name", "primary_label", "secondary_label", "window_years"])

        rookie_df["player_name"] = rookie_df["player_name"].astype(str).str.strip()
        rookie_df["player_name_cn"] = rookie_df["player_name"].map(infer_player_cn)
        rookie_df["player_name_norm"] = rookie_df["player_name"].astype(str).apply(normalize_name)
        rookie_df["player_name_cn_norm"] = rookie_df["player_name_cn"].astype(str).apply(normalize_name)

        for c in feature_cols:
            rookie_df[c] = pd.to_numeric(rookie_df[c], errors="coerce")
            veteran_df[c] = pd.to_numeric(veteran_df[c], errors="coerce")

        print(f"✅ 新秀样本: {rookie_df.shape}")
        print(f"✅ 历史样本: {veteran_df.shape}")
        print("🚀 服务启动成功")
        print(f"🌐 前端页面: http://127.0.0.1:8000")
        print(f"📖 API 文档: http://127.0.0.1:8000/docs")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        raise

    yield
    print("👋 服务已关闭")


# ====================== FastAPI 应用 ======================
app = FastAPI(
    title="NBA新秀球员分析系统",
    version="4.0",
    default_response_class=NanSafeJSONResponse,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html 不存在")
    return FileResponse(INDEX_HTML_PATH)


@app.get("/plots/{filename}")
async def get_plot(filename: str):
    fp = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(fp)


@app.get("/api/health", summary="健康检查")
async def health_check():
    return {
        "status": "ok",
        "rookie_rows": 0 if rookie_df is None else int(len(rookie_df)),
        "veteran_rows": 0 if veteran_df is None else int(len(veteran_df)),
        "windows_loaded": sorted(list(primary_stage1_models.keys())) if primary_stage1_models else [],
    }


@app.get("/api/rookies", summary="获取新秀列表")
async def get_rookies():
    if rookie_df is None or rookie_df.empty:
        raise HTTPException(status_code=500, detail="新秀样本未加载")

    latest = rookie_df.sort_values(["player_name", "window_years"], ascending=[True, False]).drop_duplicates("player_name", keep="first").copy()
    latest["analysis_stage"] = latest["window_years"].map({1: "早期观察", 2: "中期判断", 3: "正式预测"}).fillna("早期观察")
    latest["confidence_level"] = latest["window_years"].map({1: "低", 2: "中", 3: "高"}).fillna("低")

    rows = []
    for _, r in latest.iterrows():
        rows.append({
            "player_name": r["player_name"],
            "player_name_cn": r["player_name_cn"],
            "player_position": r.get("position_mode", "未知"),
            "window_years": safe_int(r.get("window_years"), 1),
            "analysis_stage": r["analysis_stage"],
            "confidence_level": r["confidence_level"],
            "newbie_age": round(safe_float(r.get("age_mean")), 1),
            "early_avg_score": round(safe_float(r.get("points_per36")), 1),
        })
    return rows


@app.post("/api/predict", summary="球员预测")
async def predict_player(request: PredictRequest):
    if rookie_df is None or rookie_df.empty:
        raise HTTPException(status_code=500, detail="新秀样本未加载")

    matches = find_player_rows_by_query(request.player_name)
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"未找到球员：{request.player_name}")

    if request.window_years is not None:
        matches = matches[matches["window_years"] == request.window_years].copy()
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"该球员不存在 window={request.window_years} 的样本")

    row = matches.sort_values("window_years", ascending=False).iloc[0]
    return build_prediction_payload(row)


@app.post("/api/players/compare", summary="球员横向对比")
async def compare_players(request: CompareRequest):
    a = await predict_player(PredictRequest(player_name=request.player_name_a))
    b = await predict_player(PredictRequest(player_name=request.player_name_b))
    return {"player_a": a, "player_b": b}


@app.get("/api/feature-importance", summary="获取一级主结果 SHAP 特征重要性")
async def get_feature_importance():
    candidates = [
        os.path.join(RUNTIME_DIR, "shap_outputs", "primary_stage1_window_3_global_shap_importance.csv"),
        os.path.join(RUNTIME_DIR, "primary_stage1_window_3_global_shap_importance.csv"),
    ]
    fp = None
    for c in candidates:
        if os.path.exists(c):
            fp = c
            break
    if fp is None:
        raise HTTPException(status_code=404, detail="未找到 window=3 的 SHAP 重要性文件")

    df = pd.read_csv(fp).fillna(0)
    df = df.sort_values("importance", ascending=False).head(10)
    return {
        "top_features": df.to_dict(orient="records"),
        "core_feature": df.iloc[0]["feature"] if not df.empty else "未知"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")
