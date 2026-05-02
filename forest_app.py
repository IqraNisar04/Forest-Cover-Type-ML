import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, f1_score,
    precision_score, recall_score, silhouette_score
)
from sklearn.preprocessing import label_binarize

st.set_page_config(
    page_title="Forest Cover Type Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg,
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color:
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .cover-card {
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .metric-box {
        background:
        border: 1px solid
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .winner-badge {
        background: linear-gradient(135deg,
        color: white;
        border-radius: 20px;
        padding: 0.4rem 1.2rem;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .info-box {
        background:
        border-left: 4px solid
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .stSlider > div { padding-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

COVER_NAMES = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}

COVER_COLORS = {
    1: '#2d6a4f',
    2: '#40916c',
    3: '#52b788',
    4: '#74c69d',
    5: '#95d5b2',
    6: '#b7e4c7',
    7: '#1b4332'
}

COVER_EMOJI = {
    1: '🌲', 2: '🌲', 3: '🌳', 4: '🌿',
    5: '🍃', 6: '🌲', 7: '⛰️'
}

CONTINUOUS_COLS = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

st.markdown('<div class="main-title">🌲 Forest Cover Type Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CS-245 Machine Learning Project | DS-2-A | Roosevelt National Forest, Colorado</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/forest.png", width=70)
    st.title("⚙️ Controls")
    st.markdown("---")

    st.subheader("📂 Dataset")
    uploaded_file = st.file_uploader("Upload covtype.csv", type=["csv"])

    st.markdown("---")
    st.subheader("🔧 Model Settings")
    n_estimators = st.slider("RF: Number of Trees", 50, 300, 200, 50)
    test_size = st.slider("Test Split Size", 0.1, 0.3, 0.2, 0.05)
    sample_size = st.slider("Training Sample (k rows)", 10000, 100000, 50000, 10000,
                             help="Use a sample for faster training. Full dataset = 581,012 rows.")

    st.markdown("---")
    st.markdown("**CS-245: Machine Learning**")
    st.markdown("*DS-2-A | Group Project*")
    st.markdown("Ayesha | Sara | Iqra | Aneeqa")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Predict", "🏋️ Train Models", "📊 Compare Models",
    "🔍 Clustering", "ℹ️ About"
])

with tab1:
    st.subheader("🌲 Predict Forest Cover Type")
    st.markdown("""
    <div class='info-box'>
    Enter the cartographic measurements for a 30×30 meter land cell. 
    The model will predict its forest cover type. Train the model first in the <b>Train Models</b> tab!
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("#### 📐 Continuous Features")
        c1, c2 = st.columns(2)
        with c1:
            elevation = st.number_input("Elevation (m)", 1500, 4000, 2800,
                                         help="Elevation in meters above sea level")
            aspect = st.number_input("Aspect (°)", 0, 360, 180,
                                      help="Compass direction the slope faces (0-360°)")
            slope = st.number_input("Slope (°)", 0, 70, 15,
                                     help="Steepness of slope in degrees")
            h_dist_hydro = st.number_input("H. Distance to Hydrology (m)", 0, 1500, 300)
            v_dist_hydro = st.number_input("V. Distance to Hydrology (m)", -200, 700, 50)
        with c2:
            h_dist_road = st.number_input("H. Distance to Roadways (m)", 0, 7000, 1500)
            hillshade_9am = st.number_input("Hillshade 9am (0-255)", 0, 255, 200)
            hillshade_noon = st.number_input("Hillshade Noon (0-255)", 0, 255, 220)
            hillshade_3pm = st.number_input("Hillshade 3pm (0-255)", 0, 255, 150)
            h_dist_fire = st.number_input("H. Distance to Fire Points (m)", 0, 7500, 1500)

        st.markdown("#### 🏕️ Wilderness Area")
        wilderness = st.radio("Select Wilderness Area:",
                               ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"],
                               horizontal=True)

        st.markdown("#### 🪨 Soil Type")
        soil_type = st.selectbox("Select Soil Type:", [f"Soil Type {i}" for i in range(1, 41)])

        submitted = st.form_submit_button("🔮 Predict Cover Type", use_container_width=True)

    if submitted:
        if 'rf_model' not in st.session_state:
            st.error("⚠️ Please train the model first in the **Train Models** tab!")
        else:
            wilderness_map = {"Rawah": 0, "Neota": 1, "Comanche Peak": 2, "Cache la Poudre": 3}
            soil_num = int(soil_type.split()[-1]) - 1

            cont_vals = [elevation, aspect, slope, h_dist_hydro, v_dist_hydro,
                         h_dist_road, hillshade_9am, hillshade_noon, hillshade_3pm, h_dist_fire]

            wilderness_vec = [0, 0, 0, 0]
            wilderness_vec[wilderness_map[wilderness]] = 1

            soil_vec = [0] * 40
            soil_vec[soil_num] = 1

            feature_vec = np.array(cont_vals + wilderness_vec + soil_vec).reshape(1, -1)
            feature_df = pd.DataFrame(feature_vec, columns=st.session_state['feature_cols'])

            scaler = st.session_state['scaler']
            feature_scaled = feature_df.copy()
            feature_scaled[CONTINUOUS_COLS] = scaler.transform(feature_df[CONTINUOUS_COLS])

            rf = st.session_state['rf_model']
            lr = st.session_state['lr_model']

            rf_pred = rf.predict(feature_scaled)[0]
            lr_pred = lr.predict(feature_scaled)[0]
            rf_proba = rf.predict_proba(feature_scaled)[0]
            lr_proba = lr.predict_proba(feature_scaled)[0]

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            col_rf, col_lr = st.columns(2)
            with col_rf:
                st.markdown(f"""
                <div class='cover-card' style='background: linear-gradient(135deg, #2d6a4f, #40916c);'>
                    🌲 Random Forest (Best Model)<br>
                    <span style='font-size:1.8rem'>{COVER_EMOJI[rf_pred]} {COVER_NAMES[rf_pred]}</span><br>
                    <span style='font-size:0.9rem; opacity:0.85'>Confidence: {rf_proba[rf_pred-1]*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            with col_lr:
                st.markdown(f"""
                <div class='cover-card' style='background: linear-gradient(135deg, #1d3461, #1f5c99);'>
                    📊 Logistic Regression (Baseline)<br>
                    <span style='font-size:1.8rem'>{COVER_EMOJI[lr_pred]} {COVER_NAMES[lr_pred]}</span><br>
                    <span style='font-size:0.9rem; opacity:0.85'>Confidence: {lr_proba[lr_pred-1]*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📊 Random Forest Probability Breakdown")
            classes_sorted = sorted(COVER_NAMES.keys())
            proba_df = pd.DataFrame({
                'Cover Type': [f"{COVER_EMOJI[c]} {COVER_NAMES[c]}" for c in classes_sorted],
                'Probability': rf_proba
            }).sort_values('Probability', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors_bar = ['#2d6a4f' if i == 0 else '#95d5b2' for i in range(7)]
            bars = ax.barh(proba_df['Cover Type'], proba_df['Probability'],
                           color=colors_bar, edgecolor='white')
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Probabilities (Random Forest)', fontweight='bold')
            ax.set_xlim(0, 1)
            for bar, val in zip(bars, proba_df['Probability']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with tab2:
    st.subheader("🏋️ Train All Three Models")

    if uploaded_file is None:
        st.markdown("""
        <div class='info-box'>
        👈 Upload <b>covtype.csv</b> in the sidebar to begin. 
        The file can be large — use the sample size slider to control training speed.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📌 Expected Dataset Format")
        st.code("""Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology,
Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways,
Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
Horizontal_Distance_To_Fire_Points,
Wilderness_Area1–4 (binary),
Soil_Type1–40 (binary),
Cover_Type (1–7 target)""")
    else:
        try:
            with st.spinner("Loading dataset..."):
                df = pd.read_csv(uploaded_file)
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)

            st.success(f"✅ Dataset loaded: {len(df):,} rows, {df.shape[1]} columns")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Samples", f"{len(df):,}")
            c2.metric("Features", f"{df.shape[1]-1}")
            c3.metric("Cover Types", "7")
            c4.metric("Missing Values", str(df.isnull().sum().sum()))

            if st.button("⚡ Train All Models Now", type="primary", use_container_width=True):

                X = df.drop('Cover_Type', axis=1)
                y = df['Cover_Type']
                continuous_cols_present = [c for c in CONTINUOUS_COLS if c in X.columns]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                scaler = StandardScaler()
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                X_train_scaled[continuous_cols_present] = scaler.fit_transform(X_train[continuous_cols_present])
                X_test_scaled[continuous_cols_present] = scaler.transform(X_test[continuous_cols_present])

                with st.spinner("Training Logistic Regression..."):
                    t0 = time.time()
                    lr = LogisticRegression(solver='lbfgs',
                                            max_iter=1000, class_weight='balanced',
                                            C=1.0, random_state=42, n_jobs=-1)
                    lr.fit(X_train_scaled, y_train)
                    lr_time = time.time() - t0
                st.success(f"✅ Logistic Regression trained in {lr_time:.1f}s")

                with st.spinner(f"Training Random Forest ({n_estimators} trees)..."):
                    t0 = time.time()
                    rf = RandomForestClassifier(n_estimators=n_estimators,
                                                class_weight='balanced',
                                                random_state=42, n_jobs=-1)
                    rf.fit(X_train_scaled, y_train)
                    rf_time = time.time() - t0
                st.success(f"✅ Random Forest trained in {rf_time:.1f}s")

                with st.spinner("Training K-Means (k=7) with PCA..."):
                    cont_only = scaler.transform(X[continuous_cols_present]) if len(X) > 0 else None
                    pca_km = PCA(n_components=2, random_state=42)
                    X_pca = pca_km.fit_transform(cont_only)
                    sample_idx = np.random.choice(len(X_pca), min(30000, len(X_pca)), replace=False)
                    km = KMeans(n_clusters=7, random_state=42, n_init=10)
                    km.fit(X_pca)
                    cluster_labels = km.predict(X_pca)
                    sil = silhouette_score(X_pca[sample_idx], cluster_labels[sample_idx])
                st.success(f"✅ K-Means trained. Silhouette Score: {sil:.4f}")

                y_pred_lr = lr.predict(X_test_scaled)
                y_pred_rf = rf.predict(X_test_scaled)

                st.session_state.update({
                    'lr_model': lr, 'rf_model': rf, 'km_model': km,
                    'scaler': scaler, 'pca_km': pca_km,
                    'X_pca': X_pca, 'cluster_labels': cluster_labels,
                    'y': y, 'X': X,
                    'y_test': y_test, 'y_pred_lr': y_pred_lr, 'y_pred_rf': y_pred_rf,
                    'feature_cols': list(X.columns),
                    'lr_time': lr_time, 'rf_time': rf_time, 'sil': sil,
                    'classes': sorted(y.unique()),
                    'class_names': [COVER_NAMES[i] for i in sorted(y.unique())],
                    'continuous_cols_present': continuous_cols_present,
                    'df': df
                })

                st.markdown("---")
                st.markdown("### 📋 Quick Results")
                ca, cb, cc, cd = st.columns(4)
                ca.metric("🌲 RF Accuracy", f"{accuracy_score(y_test, y_pred_rf)*100:.1f}%")
                cb.metric("📊 LR Accuracy", f"{accuracy_score(y_test, y_pred_lr)*100:.1f}%")
                cc.metric("🌲 RF Macro F1", f"{f1_score(y_test, y_pred_rf, average='macro'):.3f}")
                cd.metric("🔵 K-Means Silhouette", f"{sil:.3f}")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

with tab3:
    st.subheader("📊 Model Comparison & Evaluation")

    if 'rf_model' not in st.session_state:
        st.info("Train the models first in the **Train Models** tab.")
    else:
        y_test = st.session_state['y_test']
        y_pred_lr = st.session_state['y_pred_lr']
        y_pred_rf = st.session_state['y_pred_rf']
        class_names = st.session_state['class_names']
        classes = st.session_state['classes']

        st.markdown("#### 📈 Overall Metrics Comparison")
        metrics = {
            'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall',
                       'Macro F1-Score ⭐', 'Weighted F1', 'Training Time (s)'],
            'Logistic Regression': [
                f"{accuracy_score(y_test, y_pred_lr)*100:.2f}%",
                f"{precision_score(y_test, y_pred_lr, average='macro', zero_division=0):.3f}",
                f"{recall_score(y_test, y_pred_lr, average='macro'):.3f}",
                f"{f1_score(y_test, y_pred_lr, average='macro'):.3f}",
                f"{f1_score(y_test, y_pred_lr, average='weighted'):.3f}",
                f"{st.session_state['lr_time']:.1f}s"
            ],
            'Random Forest 🏆': [
                f"{accuracy_score(y_test, y_pred_rf)*100:.2f}%",
                f"{precision_score(y_test, y_pred_rf, average='macro', zero_division=0):.3f}",
                f"{recall_score(y_test, y_pred_rf, average='macro'):.3f}",
                f"{f1_score(y_test, y_pred_rf, average='macro'):.3f}",
                f"{f1_score(y_test, y_pred_rf, average='weighted'):.3f}",
                f"{st.session_state['rf_time']:.1f}s"
            ]
        }
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)

        st.markdown("#### 🗂️ Confusion Matrices")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        for ax, y_pred, title, cmap in zip(
            axes, [y_pred_lr, y_pred_rf],
            ['Logistic Regression', 'Random Forest 🏆'],
            ['Blues', 'Greens']
        ):
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm.round(2),
                                          display_labels=class_names)
            disp.plot(ax=ax, colorbar=True, cmap=cmap, xticks_rotation=40)
            ax.set_title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### 🎯 Per-Class F1 Scores")
        f1_lr_vals = f1_score(y_test, y_pred_lr, average=None)
        f1_rf_vals = f1_score(y_test, y_pred_rf, average=None)

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(class_names))
        w = 0.35
        ax.bar(x - w/2, f1_lr_vals, w, label='Logistic Regression', color='steelblue',
               alpha=0.85, edgecolor='navy')
        ax.bar(x + w/2, f1_rf_vals, w, label='Random Forest', color='#2d6a4f',
               alpha=0.85, edgecolor='#1b4332')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha='right')
        ax.set_ylabel('F1-Score')
        ax.set_ylim(0, 1.15)
        ax.set_title('Per-Class F1 Score: LR vs RF', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for bars, vals in [(ax.containers[0], f1_lr_vals), (ax.containers[1], f1_rf_vals)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### 🔑 Random Forest Feature Importance (Top 15)")
        rf = st.session_state['rf_model']
        X = st.session_state['X']
        importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)
        fig, ax = plt.subplots(figsize=(9, 5))
        importances.sort_values().plot(kind='barh', ax=ax, color='#2d6a4f',
                                        edgecolor='#1b4332', alpha=0.85)
        ax.set_title('Top 15 Feature Importances — Random Forest', fontweight='bold')
        ax.set_xlabel('Importance (Gini)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab4:
    st.subheader("🔍 K-Means Clustering (Unsupervised)")

    if 'km_model' not in st.session_state:
        st.info("Train the models first in the **Train Models** tab.")
    else:
        X_pca = st.session_state['X_pca']
        cluster_labels = st.session_state['cluster_labels']
        y = st.session_state['y']
        pca_km = st.session_state['pca_km']
        sil = st.session_state['sil']

        ca, cb = st.columns(2)
        ca.metric("Silhouette Score (k=7)", f"{sil:.4f}",
                  help="Range -1 to 1. Higher = better separated clusters.")
        cb.metric("Variance Captured by 2 PCs",
                  f"{pca_km.explained_variance_ratio_.sum()*100:.1f}%")

        sample_idx = np.random.choice(len(X_pca), min(20000, len(X_pca)), replace=False)
        X_v = X_pca[sample_idx]
        y_v = y.values[sample_idx]
        cl_v = cluster_labels[sample_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sc1 = axes[0].scatter(X_v[:, 0], X_v[:, 1], c=cl_v, cmap='tab10',
                               alpha=0.3, s=3)
        axes[0].set_title("K-Means Clusters (k=7)", fontweight='bold')
        axes[0].set_xlabel(f"PC1 ({pca_km.explained_variance_ratio_[0]*100:.1f}% var)")
        axes[0].set_ylabel(f"PC2 ({pca_km.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.colorbar(sc1, ax=axes[0], label='Cluster ID')

        sc2 = axes[1].scatter(X_v[:, 0], X_v[:, 1], c=y_v - 1, cmap='tab10',
                               alpha=0.3, s=3)
        axes[1].set_title("True Cover Types (PCA space)", fontweight='bold')
        axes[1].set_xlabel(f"PC1 ({pca_km.explained_variance_ratio_[0]*100:.1f}% var)")
        axes[1].set_ylabel(f"PC2 ({pca_km.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.colorbar(sc2, ax=axes[1], label='Cover Type')

        plt.suptitle("K-Means Clusters vs True Labels (PCA Space)", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Interpretation:** K-Means partially captures geographic patterns (high vs low elevation zones),
        but cannot cleanly separate ecologically similar species — especially Spruce/Fir and Lodgepole Pine.
        This confirms that **supervised learning is essential** for accurate forest cover classification.
        """)

with tab5:
    st.subheader("ℹ️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        
        **Dataset:** UCI Forest Cover Type (Roosevelt National Forest, Colorado)
        - 581,012 instances | 54 features | 7 target classes
        - No missing values | Severe class imbalance (105× ratio)
        
        **Problem Type:** Multi-class Classification
        
        ---
        
        | Name | CMS ID |
        |------|--------|
        | Ayesha Mobeen | 513958 |
        | Sara Fawad | 509615 |
        | Iqra Nisar | 501191 |
        | Aneeqa Zahid | 519265 |
        
        ---
        
        **Course:** CS-245 Machine Learning  
        **Class:** DS-2-A
        """)

    with col2:
        st.markdown("""
        
        **1. Logistic Regression** *(Supervised — Linear Baseline)*
        - Multinomial softmax formulation
        - class_weight='balanced', L2 regularization
        - Interpretable linear boundary
        
        **2. Random Forest** *(Supervised — Best Model 🏆)*
        - 200 decision trees (ensemble)
        - Handles non-linear boundaries natively
        - Superior performance on minority classes
        
        **3. K-Means + PCA** *(Unsupervised)*
        - Applied to 10 continuous features
        - PCA reduces to 2D for visualization
        - Confirms geographic structure in data
        
        ---
        
        """)
        for k, v in COVER_NAMES.items():
            st.markdown(f"{COVER_EMOJI[k]} **Class {k}:** {v}")

