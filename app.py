import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

# ========== Utilities ==========
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # date parsing (YYYY-MM or YYYY-MM-DD)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    # Basic required columns
    required = {"sku", "units_sold"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    # Optional columns
    for c in ["inventory_on_hand","inbound_qty","lead_time_days","moq_units",
              "safety_stock_units","reorder_point_units","coverage_months",
              "promo_flag","stockout_flag","unit_price","product_name","category","country","channel","notes"]:
        if c not in df.columns:
            df[c] = np.nan
    # Sort
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    return df

def monthly_roll_stats(s, window=3):
    rmean = s.rolling(window, min_periods=1).mean()
    rstd  = s.rolling(window, min_periods=1).std().fillna(0)
    return rmean, rstd

def compute_inventory_metrics(sku_df, lt_days=20, moq=1000, override_latest=None):
    """
    If inventory columns exist, use them. Otherwise, compute on the fly for the latest month using sidebar inputs.
    override_latest: dict like {"inventory_on_hand": x, "inbound_qty": y}
    Returns updated df and latest KPI dict.
    """
    df = sku_df.copy()
    rmean, rstd = monthly_roll_stats(df["units_sold"], window=3)
    lt_months = max(1/4, lt_days/30.0)  # min 1 week

    # If columns missing, create for whole series using simple sim (not required for demo)
    # We only need latest-month KPI; earlier rows can stay NaN if not provided.
    if df["inventory_on_hand"].isna().all():
        # fill only latest row from override_latest
        df.loc[df.index[-1], "inventory_on_hand"] = override_latest.get("inventory_on_hand", 0)
    if df["inbound_qty"].isna().all():
        df.loc[df.index[-1], "inbound_qty"] = override_latest.get("inbound_qty", 0)
    if df["lead_time_days"].isna().all():
        df["lead_time_days"] = lt_days
    if df["moq_units"].isna().all():
        df["moq_units"] = moq

    # Safety stock & ROP (series)
    df["safety_stock_units"] = np.where(
        df["safety_stock_units"].isna(),
        (0.8 * rstd * np.sqrt(lt_months) + 0.1 * rmean).round().astype(int),
        df["safety_stock_units"]
    )
    demand_lt = (rmean * lt_months).round().astype(int)
    df["reorder_point_units"] = np.where(
        df["reorder_point_units"].isna(),
        (df["safety_stock_units"] + demand_lt).astype(int),
        df["reorder_point_units"]
    )
    # Coverage months (using rolling avg)
    avg_sales = rmean.replace(0, 1)
    df["coverage_months"] = np.where(
        df["coverage_months"].isna(),
        (df["inventory_on_hand"].fillna(0) / avg_sales).round(2),
        df["coverage_months"]
    )
    # Flags + recommended qty
    df["reorder_flag"] = np.where(
        (df["inventory_on_hand"].fillna(0) < df["reorder_point_units"].fillna(0)), 1, 0
    )
    moq_series = df["moq_units"].replace(0, moq).fillna(moq)
    raw_qty = (df["reorder_point_units"].fillna(0) + moq_series - df["inventory_on_hand"].fillna(0)).clip(lower=0)
    df["reorder_qty_recommended"] = ((raw_qty / moq_series).apply(np.ceil) * moq_series).astype(int)

    latest = df.iloc[-1]
    kpi = {
        "date": latest["date"],
        "units_sold": int(latest["units_sold"]),
        "inventory_on_hand": int(latest["inventory_on_hand"]) if pd.notna(latest["inventory_on_hand"]) else 0,
        "inbound_qty": int(latest["inbound_qty"]) if pd.notna(latest["inbound_qty"]) else 0,
        "lead_time_days": int(latest["lead_time_days"]) if pd.notna(latest["lead_time_days"]) else lt_days,
        "moq_units": int(latest["moq_units"]) if pd.notna(latest["moq_units"]) else moq,
        "safety_stock_units": int(latest["safety_stock_units"]) if pd.notna(latest["safety_stock_units"]) else 0,
        "reorder_point_units": int(latest["reorder_point_units"]) if pd.notna(latest["reorder_point_units"]) else 0,
        "coverage_months": float(latest["coverage_months"]) if pd.notna(latest["coverage_months"]) else 0.0,
        "reorder_flag": int(latest["reorder_flag"]) if pd.notna(latest["reorder_flag"]) else 0,
        "reorder_qty_recommended": int(latest["reorder_qty_recommended"]) if pd.notna(latest["reorder_qty_recommended"]) else 0,
    }
    return df, kpi

def forecast_series(df, horizon=3):
    """
    Try Prophet -> SARIMAX -> Naive MA(3)
    Input df must have columns: date (timestamp monthly), units_sold
    Returns: forecast_df with columns [date, yhat, yhat_lower, yhat_upper]
    """
    hist = df[["date","units_sold"]].dropna().copy()
    hist = hist.groupby("date", as_index=False)["units_sold"].sum()
    hist = hist.sort_values("date")

    # Prepare monthly continuity
    idx = pd.period_range(hist["date"].min(), hist["date"].max(), freq="M").to_timestamp("M")
    hist = hist.set_index("date").reindex(idx).fillna(method="ffill").reset_index().rename(columns={"index":"date"})

    # Try Prophet
    try:
        from prophet import Prophet
        p = Prophet(weekly_seasonality=False, daily_seasonality=False, seasonality_mode="additive")
        # yearly seasonality captures festivities (Tet, winter)
        p.add_seasonality(name="yearly", period=365.25, fourier_order=6)
        dfp = hist.rename(columns={"date":"ds","units_sold":"y"})
        p.fit(dfp)
        future = p.make_future_dataframe(periods=horizon, freq="M")
        fcst = p.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
        fcst = fcst.tail(horizon).rename(columns={"ds":"date"})
        return fcst
    except Exception as e1:
        try:
            # SARIMAX fallback
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y = hist["units_sold"].astype(float)
            # Simple seasonal order for monthly data
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=horizon)
            mean = pred.predicted_mean.values
            conf = pred.conf_int(alpha=0.2).values  # 80% band
            dates = pd.date_range(hist["date"].max() + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
            return pd.DataFrame({
                "date": dates,
                "yhat": mean,
                "yhat_lower": conf[:,0],
                "yhat_upper": conf[:,1],
            })
        except Exception as e2:
            # Naive moving average
            avg = hist["units_sold"].tail(3).mean()
            dates = pd.date_range(hist["date"].max() + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
            return pd.DataFrame({
                "date": dates,
                "yhat": [avg]*horizon,
                "yhat_lower": [avg*0.85]*horizon,
                "yhat_upper": [avg*1.15]*horizon,
            })

def ai_verified_badge(on_hand, inbound_next, forecast_sum):
    """
    A simple supply adequacy rule for Buyer badge:
    If on_hand + inbound for next month >= 95% of next-month forecast -> Verified
    """
    return (on_hand + inbound_next) >= 0.95 * forecast_sum

# ========== App ==========
st.set_page_config(page_title="AI Demand Forecasting Demo", layout="wide")
st.title("📈 AI Demand Forecasting + Inventory Alerts (Processed Food & Agriculture)")

with st.sidebar:
    st.header("1) Tải CSV")
    st.caption("Dùng file có sẵn: *all_skus_with_inventory_2024_2025.csv* hoặc *all_skus_sales_2024_2025.csv*")
    file = st.file_uploader("Upload CSV", type=["csv"])

    st.header("2) Cấu hình Forecast")
    horizon = st.slider("Số tháng dự báo", 1, 6, 3)

    st.header("3) Nếu CSV **không** có tồn kho (nhập tay cho kỳ gần nhất)")
    manual_onhand = st.number_input("Inventory on-hand (kỳ hiện tại)", min_value=0, value=2000, step=100)
    manual_inbound = st.number_input("Inbound (open PO) tháng tới", min_value=0, value=1000, step=100)
    manual_lt_days = st.number_input("Lead time (days)", min_value=1, value=20, step=1)
    manual_moq = st.number_input("MOQ (units)", min_value=1, value=1000, step=10)

if not file:
    st.info("⬅️ Hãy upload CSV để bắt đầu. Gợi ý: dùng file *all_skus_with_inventory_2024_2025.csv* bạn đã tải.")
    st.stop()

# Load & select SKU
try:
    df = load_csv(file)
except Exception as e:
    st.error(f"Lỗi đọc CSV: {e}")
    st.stop()

# SKU picker
skus = sorted(df["sku"].dropna().unique())
selected_sku = st.selectbox("Chọn SKU", skus)
sku_df = df[df["sku"] == selected_sku].sort_values("date")

colL, colR = st.columns([1,1])

with colL:
    st.subheader("Dữ liệu gần đây")
    st.dataframe(
        sku_df.tail(12)[
            ["date","sku","product_name","units_sold","inventory_on_hand",
             "inbound_qty","lead_time_days","moq_units","coverage_months"]
        ],
        hide_index=True,
        use_container_width=True
    )

# Forecast
fcst = forecast_series(sku_df, horizon=horizon)
fcst["date"] = pd.to_datetime(fcst["date"])

# Simple plot (matplotlib or plotly)
import plotly.express as px
import plotly.graph_objects as go

hist = sku_df[["date","units_sold"]].groupby("date", as_index=False).sum()
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["units_sold"], mode="lines+markers", name="Lịch sử"))
fig.add_trace(go.Scatter(x=fcst["date"], y=fcst["yhat"], mode="lines+markers", name="Dự báo"))
fig.add_trace(go.Scatter(
    x=pd.concat([fcst["date"], fcst["date"][::-1]]),
    y=pd.concat([fcst["yhat_upper"], fcst["yhat_lower"][::-1]]),
    fill='toself', opacity=0.15, line=dict(width=0), name="Khoảng tin cậy"
))
fig.update_layout(title="Sales History & Forecast", xaxis_title="Month", yaxis_title="Units")
st.plotly_chart(fig, use_container_width=True)

# Inventory KPIs / Alerts
has_inventory_cols = not sku_df["inventory_on_hand"].isna().all()
override_latest = {"inventory_on_hand": manual_onhand, "inbound_qty": manual_inbound}
sku_df2, kpi = compute_inventory_metrics(
    sku_df, lt_days=int(manual_lt_days), moq=int(manual_moq), override_latest=override_latest
)

with colR:
    st.subheader("⚠️ Inventory & Reorder")
    c1, c2, c3 = st.columns(3)
    c1.metric("On-hand (hiện tại)", f"{kpi['inventory_on_hand']:,}")
    c2.metric("Safety Stock", f"{kpi['safety_stock_units']:,}")
    c3.metric("Reorder Point", f"{kpi['reorder_point_units']:,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Coverage (months)", f"{kpi['coverage_months']:.2f}")
    c5.metric("Lead time (days)", f"{kpi['lead_time_days']}")
    c6.metric("MOQ", f"{kpi['moq_units']:,}")

    if kpi["reorder_flag"] == 1:
        st.error(f"⚠️ Cảnh báo: On-hand ({kpi['inventory_on_hand']:,}) < ROP ({kpi['reorder_point_units']:,}). "
                 f"Đề xuất đặt hàng: **{kpi['reorder_qty_recommended']:,}** (MOQ {kpi['moq_units']:,}).")
    elif kpi["coverage_months"] < 1.0:
        st.warning(f"⚠️ Coverage thấp: chỉ **{kpi['coverage_months']:.2f}** tháng tồn kho.")
    else:
        st.success("✅ Tồn kho an toàn theo ROP & coverage.")

# Inventory vs ROP chart
inv_fig = go.Figure()
inv_fig.add_trace(go.Scatter(
    x=sku_df2["date"], y=sku_df2["inventory_on_hand"], mode="lines+markers", name="On-hand"
))
inv_fig.add_trace(go.Scatter(
    x=sku_df2["date"], y=sku_df2["reorder_point_units"], mode="lines+markers", name="Reorder Point"
))
inv_fig.update_layout(title="Inventory vs Reorder Point", xaxis_title="Month", yaxis_title="Units")
st.plotly_chart(inv_fig, use_container_width=True)

# Buyer "AI Verified Supply" badge (optional)
# Rule: if next-month forecast can be met by on_hand + inbound -> Verified
next_month_need = float(fcst.iloc[0]["yhat"]) if len(fcst) > 0 else 0.0
verified = ai_verified_badge(kpi["inventory_on_hand"], kpi["inbound_qty"], next_month_need)

st.markdown("---")
st.subheader("Buyer Experience (Badge)")
if verified:
    st.success("🔮 **AI Verified Supply**: Nguồn cung ổn định cho tháng tới.")
else:
    st.info("ℹ️ Chưa đạt tiêu chí **AI Verified Supply** cho tháng tới.")
st.caption("Quy tắc demo: on-hand + inbound ≥ 95% nhu cầu dự báo tháng tới.")
