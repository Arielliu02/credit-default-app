import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 頁面設定 ────────────────────────────────────────────
st.set_page_config(
    page_title='信用卡違約預警系統',
    page_icon='🚨',
    layout='wide'
)

# ── 自訂 CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }

.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1e2130, #252840);
    border: 1px solid #3a3f5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
}
.metric-label { color: #8892b0; font-size: 13px; margin-bottom: 6px; }
.metric-value { color: #ccd6f6; font-size: 28px; font-weight: 700; }

.alert-high {
    background: linear-gradient(135deg, #3d0000, #5c1a1a);
    border: 2px solid #ff4444;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.alert-low {
    background: linear-gradient(135deg, #003d1a, #1a5c35);
    border: 2px solid #44ff88;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.alert-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
.alert-subtitle { font-size: 14px; opacity: 0.8; }

.section-title {
    color: #64ffda;
    font-size: 18px;
    font-weight: 700;
    border-bottom: 2px solid #64ffda33;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.sidebar-label { color: #8892b0; font-size: 12px; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)


# ── 訓練模型（快取）────────────────────────────────────
@st.cache_resource
def train_model(filepath):
    df = pd.read_excel(filepath, header=1)
    df = df.rename(columns={
        'LIMIT_BAL':'X1','SEX':'X2','EDUCATION':'X3','MARRIAGE':'X4','AGE':'X5',
        'PAY_0':'X6','PAY_2':'X7','PAY_3':'X8','PAY_4':'X9','PAY_5':'X10','PAY_6':'X11',
        'BILL_AMT1':'X12','BILL_AMT2':'X13','BILL_AMT3':'X14',
        'BILL_AMT4':'X15','BILL_AMT5':'X16','BILL_AMT6':'X17',
        'PAY_AMT1':'X18','PAY_AMT2':'X19','PAY_AMT3':'X20',
        'PAY_AMT4':'X21','PAY_AMT5':'X22','PAY_AMT6':'X23',
        'default payment next month':'Y'
    })
    df = df.drop(columns=['ID'])
    df['X3'] = df['X3'].replace({0:4,5:4,6:4})
    df['X4'] = df['X4'].replace({0:3})
    for c in ['X6','X7','X8','X9','X10','X11']:
        df[c] = df[c].replace({-2:0,-1:0})

    X = df.drop(columns=['Y'])
    y = df['Y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    neg, pos = (y_train==0).sum(), (y_train==1).sum()
    model = XGBClassifier(
        n_estimators=500, random_state=42, eval_metric='logloss',
        scale_pos_weight=neg/pos, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns.tolist()


# ── 側邊欄 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('## ⚙️ 系統設定')
    filepath = st.text_input(
        '資料集路徑',
        value='default of credit card clients.xls'
    )
    threshold = st.slider('預測門檻（Threshold）', 0.1, 0.7, 0.35, 0.01,
                          help='降低門檻 → 更積極預測違約（Recall↑，Precision↓）')
    st.markdown('---')
    st.markdown('## 📋 客戶資料輸入')

    col1, col2 = st.columns(2)
    with col1:
        limit_bal = st.number_input('信用額度（NT$）', 10000, 1000000, 200000, 10000)
        age       = st.number_input('年齡', 18, 80, 35)
        sex       = st.selectbox('性別', [('男', 1), ('女', 2)], format_func=lambda x: x[0])
        edu       = st.selectbox('教育程度',
                                 [('研究所',1),('大學',2),('高中',3),('其他',4)],
                                 format_func=lambda x: x[0])
    with col2:
        mar       = st.selectbox('婚姻狀況',
                                 [('已婚',1),('單身',2),('其他',3)],
                                 format_func=lambda x: x[0])
        pay_sep   = st.selectbox('9月還款狀態',
                                 [('正常',0),('延遲1月',1),('延遲2月',2),('延遲3月',3)],
                                 format_func=lambda x: x[0])
        pay_aug   = st.selectbox('8月還款狀態',
                                 [('正常',0),('延遲1月',1),('延遲2月',2),('延遲3月',3)],
                                 format_func=lambda x: x[0])
        pay_jul   = st.selectbox('7月還款狀態',
                                 [('正常',0),('延遲1月',1),('延遲2月',2),('延遲3月',3)],
                                 format_func=lambda x: x[0])

    st.markdown('**帳單金額（NT$）**')
    c1, c2, c3 = st.columns(3)
    bill_sep = c1.number_input('9月', 0, 1000000, 50000, 1000)
    bill_aug = c2.number_input('8月', 0, 1000000, 48000, 1000)
    bill_jul = c3.number_input('7月', 0, 1000000, 46000, 1000)
    bill_jun = c1.number_input('6月', 0, 1000000, 44000, 1000)
    bill_may = c2.number_input('5月', 0, 1000000, 42000, 1000)
    bill_apr = c3.number_input('4月', 0, 1000000, 40000, 1000)

    st.markdown('**繳款金額（NT$）**')
    c1, c2, c3 = st.columns(3)
    pay_a1 = c1.number_input('9月 ', 0, 500000, 2000, 500)
    pay_a2 = c2.number_input('8月 ', 0, 500000, 2000, 500)
    pay_a3 = c3.number_input('7月 ', 0, 500000, 2000, 500)
    pay_a4 = c1.number_input('6月 ', 0, 500000, 2000, 500)
    pay_a5 = c2.number_input('5月 ', 0, 500000, 2000, 500)
    pay_a6 = c3.number_input('4月 ', 0, 500000, 2000, 500)

    predict_btn = st.button('🔍 執行預測', use_container_width=True, type='primary')


# ── 主畫面 ──────────────────────────────────────────────
st.markdown('# 🚨 信用卡違約預警系統')
st.markdown('*基於 XGBoost 模型｜輔仁大學金融大數據期末報告*')
st.markdown('---')

# 載入模型
with st.spinner('模型載入中...'):
    try:
        model, X_test, y_test, feature_cols = train_model(filepath)
        model_loaded = True
    except Exception as e:
        st.error(f'資料載入失敗：{e}')
        model_loaded = False

if model_loaded:
    y_prob_all = model.predict_proba(X_test)[:, 1]
    y_pred_all = (y_prob_all >= threshold).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc  = accuracy_score(y_test, y_pred_all)
    prec = precision_score(y_test, y_pred_all)
    rec  = recall_score(y_test, y_pred_all)
    f1   = f1_score(y_test, y_pred_all)
    auc  = roc_auc_score(y_test, y_prob_all)

    # ── 模型績效 KPI ──────────────────────────────────────
    st.markdown('<div class="section-title">📊 模型整體績效（測試集）</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, label, val in zip(cols,
        ['Accuracy','Precision','Recall','F1 Score','AUC'],
        [acc, prec, rec, f1, auc]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── 圖表區 ────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(['📈 ROC 曲線', '🔲 混淆矩陣', '🏆 特徵重要性'])

    with tab1:
        fpr, tpr, _ = roc_curve(y_test, y_prob_all)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        ax.plot(fpr, tpr, color='#64ffda', lw=2.5, label=f'XGBoost (AUC={auc:.4f})')
        ax.plot([0,1],[0,1],'--', color='#8892b0', lw=1)
        ax.set_xlabel('False Positive Rate', color='#ccd6f6')
        ax.set_ylabel('True Positive Rate', color='#ccd6f6')
        ax.set_title('ROC 曲線', color='#ccd6f6', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3f5c')
        ax.legend(facecolor='#252840', edgecolor='#3a3f5c', labelcolor='#ccd6f6')
        ax.grid(alpha=0.15, color='#3a3f5c')
        st.pyplot(fig)

    with tab2:
        cm = confusion_matrix(y_test, y_pred_all)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['未違約','違約'],
                    yticklabels=['未違約','違約'],
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title('混淆矩陣（測試集）', color='#ccd6f6', fontsize=12, fontweight='bold')
        ax.set_xlabel('預測值', color='#ccd6f6')
        ax.set_ylabel('實際值', color='#ccd6f6')
        ax.tick_params(colors='#ccd6f6')
        st.pyplot(fig)

    with tab3:
        importance = model.feature_importances_
        feat_df = pd.DataFrame({'特徵': feature_cols, '重要性': importance})
        feat_df = feat_df.sort_values('重要性', ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        bars = ax.barh(feat_df['特徵'], feat_df['重要性'], color='#64ffda', alpha=0.85)
        ax.set_title('Top 15 特徵重要性', color='#ccd6f6', fontsize=13, fontweight='bold')
        ax.set_xlabel('重要性分數', color='#ccd6f6')
        ax.tick_params(colors='#ccd6f6')
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3f5c')
        ax.grid(axis='x', alpha=0.15, color='#3a3f5c')
        st.pyplot(fig)

    st.markdown('---')

    # ── 單筆預測 ──────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 個別客戶違約預測</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame([[
            limit_bal, sex[1], edu[1], mar[1], age,
            pay_sep[1], pay_aug[1], pay_jul[1], 0, 0, 0,
            bill_sep, bill_aug, bill_jul, bill_jun, bill_may, bill_apr,
            pay_a1, pay_a2, pay_a3, pay_a4, pay_a5, pay_a6
        ]], columns=feature_cols)

        prob  = model.predict_proba(input_data)[0][1]
        pred  = int(prob >= threshold)
        pct   = prob * 100

        col_a, col_b = st.columns([1, 2])

        with col_a:
            if pred == 1:
                st.markdown(f"""
                <div class="alert-high">
                    <div class="alert-title" style="color:#ff6b6b;">⚠️ 高違約風險</div>
                    <div style="font-size:42px;font-weight:900;color:#ff4444;">{pct:.1f}%</div>
                    <div class="alert-subtitle" style="color:#ffaaaa;">違約機率｜建議加強審查</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-low">
                    <div class="alert-title" style="color:#6bffb8;">✅ 低違約風險</div>
                    <div style="font-size:42px;font-weight:900;color:#44ff88;">{pct:.1f}%</div>
                    <div class="alert-subtitle" style="color:#aaffcc;">違約機率｜客戶信用正常</div>
                </div>""", unsafe_allow_html=True)

        with col_b:
            # 機率儀表條
            fig, ax = plt.subplots(figsize=(6, 1.5), facecolor='#1e2130')
            ax.set_facecolor('#1e2130')
            ax.barh([0], [1], color='#2a2f45', height=0.5)
            bar_color = '#ff4444' if pred == 1 else '#44ff88'
            ax.barh([0], [prob], color=bar_color, height=0.5, alpha=0.9)
            ax.axvline(threshold, color='#ffdd57', lw=2, linestyle='--')
            ax.text(threshold, 0.35, f'門檻 {threshold}', color='#ffdd57',
                    fontsize=9, ha='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.8)
            ax.axis('off')
            ax.set_title('違約機率儀表', color='#ccd6f6', fontsize=11)
            st.pyplot(fig)

            # 輸入摘要
            st.markdown('**📋 輸入摘要**')
            summary = {
                '信用額度': f'NT$ {limit_bal:,}',
                '年齡': f'{age} 歲',
                '性別': sex[0],
                '教育程度': edu[0],
                '婚姻狀況': mar[0],
                '9月還款狀態': pay_sep[0],
            }
            st.dataframe(pd.DataFrame(summary.items(), columns=['項目','數值']),
                         use_container_width=True, hide_index=True)
    else:
        st.info('👈 請在左側填入客戶資料後，點擊「執行預測」按鈕')
