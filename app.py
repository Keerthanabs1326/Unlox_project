# app.py (full updated)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error
import os

st.set_page_config(page_title='Retail Analytics & Forecasting', layout='wide')

# ---------------------
# Helpers & caching
# ---------------------
@st.cache_data(show_spinner=False)
def load_parquet_safe(path):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def ensure_date_col(df, col='date'):
    if df is None:
        return None
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

def safe_rename_category(df):
    if df is None:
        return None
    if 'category' in df.columns and 'product_category' not in df.columns:
        df = df.rename(columns={'category': 'product_category'})
    return df

def compute_forecast_metrics(true_df, pred_df, date_col='date', true_col='revenue', pred_col='yhat'):
    """Compute MAE and MAPE between true_df and pred_df on overlapping dates."""
    if true_df is None or pred_df is None:
        return None
    # merge on date
    merged = pd.merge(true_df[[date_col, true_col]], pred_df[[date_col, pred_col]], on=date_col, how='inner')
    if merged.empty:
        return None
    mae = mean_absolute_error(merged[true_col], merged[pred_col])
    # avoid division by zero for MAPE: ignore zero true values
    nonzero_mask = merged[true_col] != 0
    if nonzero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = (np.abs((merged[true_col] - merged[pred_col]) / merged[true_col])[nonzero_mask]).mean() * 100
    return {'mae': mae, 'mape_pct': mape, 'n_points': len(merged)}

# ---------------------
# Load artifacts (cached)
# ---------------------
daily = load_parquet_safe('daily.parquet')
forecasts = load_parquet_safe('forecasts.parquet')

if daily is None:
    st.error("Missing `daily.parquet`. Please place it in the app folder and reload.")
    st.stop()

if forecasts is None:
    st.warning("`forecasts.parquet` not found — forecast charts & metrics will be hidden until it's available.")

# unify column names
daily = safe_rename_category(daily)
forecasts = safe_rename_category(forecasts) if forecasts is not None else None

# ensure date columns
daily = ensure_date_col(daily, 'date')
if forecasts is not None:
    forecasts = ensure_date_col(forecasts, 'date')

# ---------------------
# Sidebar filters
# ---------------------
st.sidebar.title('Filters')
store_ids = sorted(daily['store_id'].dropna().unique())
if not store_ids:
    st.error("No store_ids found in daily data.")
    st.stop()

store = st.sidebar.selectbox('Store', store_ids)

# categories for selected store (guard empty)
cats = daily.loc[daily['store_id'] == store, 'product_category'].dropna().unique()
if len(cats) == 0:
    st.sidebar.error("No product categories for this store.")
    st.stop()

category = st.sidebar.selectbox('Category', sorted(cats))

# date window filter (optional)
min_date = daily['date'].min()
max_date = daily['date'].max()
date_range = st.sidebar.date_input("Historical date range", [min_date, max_date])
if len(date_range) != 2:
    hist_start, hist_end = min_date, max_date
else:
    hist_start, hist_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# ---------------------
# Historical vs Forecast chart + forecast quality
# ---------------------
hist = daily[
    (daily.store_id == store) & (daily.product_category == category) &
    (daily['date'] >= hist_start) & (daily['date'] <= hist_end)
][['date', 'revenue']].sort_values('date')

st.subheader(f'History | {store} · {category}')

if hist.empty:
    st.info("No historical data for this selection.")
else:
    fig = px.line(hist, x='date', y='revenue', title=f'History | {store} · {category}', labels={'revenue':'revenue','date':'date'})
    # overlay forecast if available
    if forecasts is not None:
        fut = forecasts[
            (forecasts.store_id == store) & (forecasts.product_category == category)
        ][['date', 'yhat']].sort_values('date')
        if not fut.empty:
            fig.add_scatter(x=fut['date'], y=fut['yhat'], mode='lines', name='Forecast', line=dict(dash='dash'))
            # compute and show simple metrics if there are overlapping dates
            # Use historical period intersecting with forecast dates for metrics
            metrics = compute_forecast_metrics(hist, fut, date_col='date', true_col='revenue', pred_col='yhat')
            if metrics:
                st.markdown(f"**Forecast metrics (on overlapping dates):** MAE = {metrics['mae']:.2f} — MAPE = {metrics['mape_pct']:.2f}% — points = {metrics['n_points']}")
        else:
            st.info("No forecast available for this store/category.")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------
# Promotion impact chart
# ---------------------
if 'promotion_applied' in daily.columns:
    st.subheader("Promotion impact on revenue")
    promo_df = daily[(daily.store_id == store) & (daily.product_category == category)]
    if promo_df.empty:
        st.info("No data to compute promotion impact for this selection.")
    else:
        # avoid SettingWithCopy warnings
        promo_df = promo_df.copy()
        promo_df['promotion_applied'] = promo_df['promotion_applied'].astype('bool')
        promo = promo_df.groupby('promotion_applied', observed=True)['revenue'].agg(['mean','count']).reset_index().rename(columns={'mean':'mean_revenue','count':'n_days'})
        fig2 = px.bar(promo, x='promotion_applied', y='mean_revenue',
                      labels={'promotion_applied':'promotion_applied','mean_revenue':'mean_revenue'},
                      hover_data=['n_days'],
                      title="Promotion impact (mean revenue)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------
# Store segmentation (clusters) with elbow, silhouette, auto-suggest, download
# ---------------------
st.subheader("Store segmentation overview")

# Aggregate store-level features (use entire daily dataset)
agg = (daily.groupby('store_id')
       .agg(total_revenue=('revenue','sum'),
            avg_ticket=('revenue','mean'),
            revenue_volatility=('revenue','std'),
            units_mean=('units_sold','mean'),
            weekend_share=('is_weekend','mean'))
       .reset_index())

# fix NaNs for revenue_volatility (stores with single record)
agg['revenue_volatility'] = agg['revenue_volatility'].fillna(0)

feature_cols = ['total_revenue', 'avg_ticket', 'revenue_volatility', 'units_mean', 'weekend_share']
missing_features = [c for c in feature_cols if c not in agg.columns]
if missing_features:
    st.error(f"Missing expected aggregated features: {missing_features}")
    st.stop()

# clustering options
st.write("Clustering options")
col1, col2 = st.columns([1, 2])
with col1:
    log_transform = st.checkbox("Log-transform revenue features (recommended)", value=True)
with col2:
    max_k_user = st.slider("Max k to evaluate (elbow & silhouette)", min_value=4, max_value=12, value=8)

# Prepare feature DataFrame
X_df = agg[feature_cols].copy()
if log_transform:
    for c in ['total_revenue', 'avg_ticket']:
        if c in X_df.columns:
            X_df[c] = np.log1p(X_df[c])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)

n_stores = len(agg)
max_k = min(max_k_user, max(3, n_stores-1))
k_range = list(range(2, max_k + 1))

# Evaluate k (SSE & silhouette)
sse = []
sil_scores = []
for k in k_range:
    try:
        km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        sse.append(km_tmp.inertia_)
        if 1 < k < n_stores:
            sil_scores.append(silhouette_score(X_scaled, labels_tmp))
        else:
            sil_scores.append(np.nan)
    except Exception:
        sse.append(np.nan)
        sil_scores.append(np.nan)

metrics_df = pd.DataFrame({'k': k_range, 'sse': sse, 'silhouette': sil_scores})

# plots
fig_sse = px.line(metrics_df, x='k', y='sse', markers=True, title='Elbow chart (SSE) - lower is better', labels={'sse':'SSE (inertia)', 'k':'k'})
fig_sil = px.line(metrics_df, x='k', y='silhouette', markers=True, title='Silhouette score by k - higher is better', labels={'silhouette':'Silhouette', 'k':'k'})
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig_sse, use_container_width=True)
with c2:
    st.plotly_chart(fig_sil, use_container_width=True)

# automatic k suggestion:
# prefer k with max silhouette if available (and not nan); fallback to elbow heuristic (largest drop)
suggested_k = None
if metrics_df['silhouette'].notna().any():
    suggested_k = int(metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k'])
else:
    # simple elbow heuristic: largest relative drop in sse
    sse_vals = metrics_df['sse'].values
    if len(sse_vals) >= 2 and not np.any(np.isnan(sse_vals)):
        # compute relative drops
        drops = (sse_vals[:-1] - sse_vals[1:]) / sse_vals[:-1]
        elbow_idx = int(np.argmax(drops))
        suggested_k = int(metrics_df['k'].iloc[elbow_idx + 1])  # choose k after the largest drop
    else:
        suggested_k = 5 if 5 in k_range else k_range[0]

st.info(f"Suggested k = {suggested_k} (based on silhouette or elbow heuristic).")

# choose k interactively
default_k = suggested_k if suggested_k in k_range else (5 if 5 in k_range else k_range[len(k_range)//2])
chosen_k = st.slider("Choose number of clusters (k) to apply", min_value=min(k_range), max_value=max(k_range), value=default_k, step=1)

# compute final clustering for chosen_k
km = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
clusters = pd.DataFrame({'store_id': agg['store_id'].values, 'cluster': labels})

# Save clusters to parquet for chosen k
clusters_path = f'store_clusters_k{chosen_k}.parquet'
try:
    clusters.to_parquet(clusters_path, index=False)
except Exception:
    # continue silently if save fails
    pass

# show cluster distribution
cluster_counts = clusters['cluster'].value_counts().rename_axis('cluster').reset_index(name='count').sort_values('cluster')
fig_clusters = px.bar(cluster_counts, x='cluster', y='count', title=f'Store clusters (k={chosen_k})', labels={'count':'count','cluster':'cluster'})
st.plotly_chart(fig_clusters, use_container_width=True)

# ---------------------
# Cluster profiles (numeric only) + download button
# ---------------------
merged = agg.merge(clusters, on='store_id', how='inner')

# compute mean of features per cluster (explicit columns)
profile = merged.groupby('cluster', as_index=False)[feature_cols].mean().round(2)

# if log_transform was applied, append medians on original scale for interpretability
if log_transform:
    orig = daily.groupby('store_id').agg(total_revenue_orig=('revenue','sum'), avg_ticket_orig=('revenue','mean')).reset_index()
    merged_orig = merged.merge(orig, on='store_id', how='left')
    profile_orig = merged_orig.groupby('cluster', as_index=False)[['total_revenue_orig','avg_ticket_orig']].median().round(2)
    profile = profile.merge(profile_orig, on='cluster')

st.write("Cluster profiles (average metrics):")
st.dataframe(profile)

# download clusters CSV for chosen k
csv_bytes = clusters.to_csv(index=False).encode('utf-8')
st.download_button(
    label=f"Download cluster assignments (k={chosen_k}) as CSV",
    data=csv_bytes,
    file_name=f"store_clusters_k{chosen_k}.csv",
    mime="text/csv"
)

# Also show local saved parquet path (if file exists)
if os.path.exists(clusters_path):
    st.caption(f"Cluster parquet saved to: {os.path.abspath(clusters_path)}")

# silhouette for chosen_k
if 1 < chosen_k < n_stores:
    try:
        sil_final = silhouette_score(X_scaled, clusters['cluster'])
        st.caption(f"Silhouette score for chosen k={chosen_k}: {sil_final:.3f}")
    except Exception:
        pass
else:
    st.caption("Silhouette score not available for the chosen k (too few stores).")

# ---------------------
# Extra: show which cluster the selected store belongs to
# ---------------------
selected_cluster = clusters.loc[clusters['store_id'] == store, 'cluster']
if not selected_cluster.empty:
    st.info(f"Selected store `{store}` belongs to cluster {int(selected_cluster.iloc[0])}")
