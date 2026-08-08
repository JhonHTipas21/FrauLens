import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import load_model_artifacts, calculate_metrics
from src.explainer import SHAPExplainer

# Page setup
st.set_page_config(
    page_title="FraudLens — Detector de Fraude Explicable",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 23, 37) 0%, rgb(9, 10, 15) 90%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f3f4f6;
        margin-bottom: 15px;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .badge-medium { background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-low { background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    </style>
""", unsafe_allow_html=True)

# Path to files
MODEL_PATH = project_root / "models" / "fraud_model.joblib"
METADATA_PATH = project_root / "models" / "metadata.json"

@st.cache_resource
def load_model_resources():
    """Loads and caches the model pipeline components."""
    if not MODEL_PATH.exists():
        st.error("No se encontró el archivo del modelo entrenado. Ejecuta primero `python3 src/train.py`.")
        st.stop()
    return load_model_artifacts(MODEL_PATH)

@st.cache_data
def load_metadata():
    """Loads and caches the model metadata JSON."""
    if not METADATA_PATH.exists():
        st.error("No se encontró el archivo de metadatos. Ejecuta primero `python3 src/train.py`.")
        st.stop()
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

# Load resources
artifacts = load_model_resources()
metadata = load_metadata()

preprocessor = artifacts["preprocessor"]
hybrid_classifier = artifacts["classifier"]
feature_names = artifacts["feature_names"]

# Extract dataset for simulation
@st.cache_data
def get_simulation_pool():
    """Loads raw dataset and samples it to create a pool of normal and fraudulent transactions for audit."""
    from src.data_loader import load_dataset
    df = load_dataset()
    
    # Split using same temporal threshold
    max_time = df['Time'].max()
    min_time = df['Time'].min()
    time_span = max_time - min_time
    threshold = min_time + time_span * 0.8 # 80% train, 20% test
    
    test_df = df[df['Time'] >= threshold].copy()
    
    # Separate class 0 and 1 in the test set to sample them
    frauds = test_df[test_df['Class'] == 1]
    legits = test_df[test_df['Class'] == 0]
    
    # Sample 400 normal records + all frauds in test set
    legits_sample = legits.sample(n=min(len(legits), 400), random_state=42)
    
    pool = pd.concat([frauds, legits_sample]).sort_values(by='Time').reset_index(drop=True)
    return pool

simulation_pool = get_simulation_pool()

# Preprocess pool features for model ingestion
X_pool, y_pool = preprocessor.transform(simulation_pool)
y_probs = hybrid_classifier.predict_proba(X_pool)[:, 1]

# Add probabilities and indices to the view dataframe
simulation_pool['Probabilidad'] = y_probs
simulation_pool['Transaccion_ID'] = [f"TX_{i:04d}" for i in range(len(simulation_pool))]

# Layout Header
col_logo, col_title = st.columns([1, 12])
with col_title:
    st.title("FraudLens 🔍 — Detector de Fraude Explicable")
    st.markdown("<p style='color: #9ca3af; font-size: 1.15rem; margin-top:-15px;'>Módulo de explicabilidad de decisiones de ML para analistas de riesgo financiero</p>", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.markdown("### Configuración del Umbral")
threshold = st.sidebar.slider(
    "Umbral de Decisión (Threshold)", 
    min_value=0.01, 
    max_value=0.99, 
    value=0.50, 
    step=0.01,
    help="Define la probabilidad a partir de la cual una transacción se marca como fraude."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Costos Operativos de Negocio")
cost_fn = st.sidebar.number_input("Costo de Falso Negativo (Fraude omitido)", value=250.0, step=10.0, help="El costo financiero promedio que asume la empresa si no detecta una transacción fraudulenta.")
cost_fp = st.sidebar.number_input("Costo de Falso Positivo (Bloqueo erróneo)", value=10.0, step=1.0, help="El costo de auditar, llamar al cliente o enviar mensajes SMS cuando se bloquea una transacción legítima.")

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Rendimiento & Impacto Económico", 
    "🕵️ Auditoría de Transacciones", 
    "🌐 Explicabilidad Global"
])

# ================= TAB 1: RENDIMIENTO & METRICAS =================
with tab1:
    st.markdown("<div class='card-title'>Rendimiento Operativo en Tiempo Real</div>", unsafe_allow_html=True)
    
    # Calculate metrics for current threshold dynamically
    current_metrics = calculate_metrics(y_pool, y_probs, threshold=threshold)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current_metrics['auc_pr']:.4f}</div>
                <div class='metric-label'>AUC-PR</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current_metrics['precision']:.2f}%</div>
                <div class='metric-label'>Precisión</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current_metrics['recall']:.2f}%</div>
                <div class='metric-label'>Recall (Sensibilidad)</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{current_metrics['f1']:.4f}</div>
                <div class='metric-label'>F1-Score</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_plot, col_impact = st.columns([2, 1])
    
    with col_plot:
        st.markdown("<div class='card-title'>Curva de Compensación Precisión vs. Recall</div>", unsafe_allow_html=True)
        
        # Plot curve using plotly from metadata
        curve_data = pd.DataFrame(metadata["threshold_curve"])
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=curve_data['threshold'], y=curve_data['precision'], name='Precisión', line=dict(color='#ef4444', width=2)))
        fig_curve.add_trace(go.Scatter(x=curve_data['threshold'], y=curve_data['recall'], name='Recall', line=dict(color='#10b981', width=2)))
        fig_curve.add_trace(go.Scatter(x=curve_data['threshold'], y=curve_data['f1'], name='F1-Score', line=dict(color='#3b82f6', width=1.5, dash='dash')))
        
        # Add vertical line for selected threshold
        fig_curve.add_vline(x=threshold, line_width=2, line_dash="dash", line_color="#ffffff", annotation_text=f"Umbral actual: {threshold:.2f}")
        
        fig_curve.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            xaxis_title="Umbral de Decisión (Threshold)",
            yaxis_title="Puntaje Métrico",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        
    with col_impact:
        st.markdown("<div class='card-title'>Cálculo de Pérdida Financiera Operativa</div>", unsafe_allow_html=True)
        
        fp_count = current_metrics['confusion_matrix']['fp']
        fn_count = current_metrics['confusion_matrix']['fn']
        tp_count = current_metrics['confusion_matrix']['tp']
        tn_count = current_metrics['confusion_matrix']['tn']
        
        cost_total_fn = fn_count * cost_fn
        cost_total_fp = fp_count * cost_fp
        total_loss = cost_total_fn + cost_total_fp
        
        # Optimal threshold calculations based on inputs
        losses_by_th = []
        thresholds_eval = np.linspace(0.01, 0.99, 99)
        for th_eval in thresholds_eval:
            m_eval = calculate_metrics(y_pool, y_probs, threshold=float(th_eval))
            fp_eval = m_eval['confusion_matrix']['fp']
            fn_eval = m_eval['confusion_matrix']['fn']
            loss_eval = (fp_eval * cost_fp) + (fn_eval * cost_fn)
            losses_by_th.append(loss_eval)
            
        optimal_idx = np.argmin(losses_by_th)
        opt_threshold = thresholds_eval[optimal_idx]
        min_loss = losses_by_th[optimal_idx]
        
        st.markdown(f"""
            *   **Casos Falsos Negativos (Fraudes Omitidos):** `{fn_count}` 
                *   Pérdida por fraudes: **${cost_total_fn:,.2f}**
            *   **Casos Falsos Positivos (Bloqueos Legítimos):** `{fp_count}`
                *   Costo operativo de auditoría: **${cost_total_fp:,.2f}**
            *   **Costo de Pérdida Total Operativa:** <span style='font-size:1.4rem; font-weight:700; color:#ef4444;'>${total_loss:,.2f}</span>
            
            ---
            
            📊 **Optimización de Costos Basada en Simulación:**
            *   El umbral que **minimiza la pérdida económica** dadas las variables actuales es **`{opt_threshold:.2f}`**.
            *   Pérdida estimada en umbral óptimo: **${min_loss:,.2f}** (Ahorro potencial: **${max(0.0, total_loss - min_loss):,.2f}**).
        """, unsafe_allow_html=True)
        
        # Render a quick bar chart comparing current loss and min loss
        fig_loss = go.Figure(data=[
            go.Bar(name='Costo Actual', x=['Pérdida Económica'], y=[total_loss], marker_color='#ef4444'),
            go.Bar(name='Costo Mínimo Optimizado', x=['Pérdida Económica'], y=[min_loss], marker_color='#10b981')
        ])
        fig_loss.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            margin=dict(l=10, r=10, t=10, b=10),
            height=180
        )
        st.plotly_chart(fig_loss, use_container_width=True)

# ================= TAB 2: AUDITORIA DE TRANSACCIONES =================
with tab2:
    st.markdown("<div class='card-title'>Listado de Transacciones en el Pool de Simulación</div>", unsafe_allow_html=True)
    st.write("Filtra las transacciones analizadas del test temporal por tier de riesgo para auditar sus explicaciones.")
    
    # Calculate labels based on selected threshold
    simulation_pool['Decision'] = np.where(simulation_pool['Probabilidad'] >= threshold, 'Fraude', 'Legítima')
    
    # Tiers classification
    def assign_tier(prob):
        if prob >= 0.70: return 'Alto Riesgo 🔴'
        elif prob >= 0.20: return 'Medio Riesgo 🟡'
        return 'Bajo Riesgo 🟢'
    
    simulation_pool['Tier'] = simulation_pool['Probabilidad'].apply(assign_tier)
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        tier_filter = st.multiselect("Filtrar por Tier de Riesgo", options=['Alto Riesgo 🔴', 'Medio Riesgo 🟡', 'Bajo Riesgo 🟢'], default=['Alto Riesgo 🔴', 'Medio Riesgo 🟡'])
    with col_filter2:
        class_filter = st.multiselect("Filtrar por Decisión del Modelo", options=['Fraude', 'Legítima'], default=['Fraude', 'Legítima'])
        
    filtered_df = simulation_pool[
        simulation_pool['Tier'].isin(tier_filter) & 
        simulation_pool['Decision'].isin(class_filter)
    ]
    
    # Show columns that are useful for analysts
    display_cols = ['Transaccion_ID', 'Time', 'Amount', 'Probabilidad', 'Decision', 'Tier', 'Class']
    st.dataframe(
        filtered_df[display_cols].rename(columns={'Class': 'Clase Real (Ground Truth)'}),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("<div class='card-title'>Explicador Individual SHAP (Fase de Auditoría Detallada)</div>", unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.warning("No hay transacciones que coincidan con los filtros seleccionados.")
    else:
        # User selects one transaction ID to audit
        selected_tx = st.selectbox(
            "Selecciona una Transacción para Inspeccionar", 
            options=filtered_df['Transaccion_ID'].tolist()
        )
        
        # Get selected row details
        tx_row = simulation_pool[simulation_pool['Transaccion_ID'] == selected_tx].iloc[0]
        tx_idx = simulation_pool[simulation_pool['Transaccion_ID'] == selected_tx].index[0]
        
        # Explain using SHAP
        explainer = SHAPExplainer(hybrid_classifier)
        explanation = explainer.explain_instance(X_pool[tx_idx], feature_names)
        
        # Display Transaction details
        col_tx_info, col_explain_plot = st.columns([1, 2])
        
        with col_tx_info:
            st.markdown(f"### Detalles de {selected_tx}")
            
            # Label details
            real_class = "Fraude (1)" if tx_row['Class'] == 1 else "Legítima (0)"
            tier_class = tx_row['Tier']
            
            st.markdown(f"""
                *   **Monto de Transacción:** `${tx_row['Amount']:.2f}`
                *   **Tiempo (Segundos):** `{tx_row['Time']:.0f}`
                *   **Clase Real (Ground Truth):** `{real_class}`
                *   **Probabilidad de Fraude:** <span style='font-size:1.25rem; font-weight:700; color:#3b82f6;'>{tx_row['Probabilidad'] * 100:.2f}%</span>
                *   **Tier de Riesgo:** `{tier_class}`
                *   **Decisión en Umbral actual:** `{"BLOQUEADO" if tx_row['Decision'] == "Fraude" else "PERMITIDO"}`
            """, unsafe_allow_html=True)
            
            # Risk recommendation text
            if tx_row['Probabilidad'] >= 0.70:
                st.error("🚨 **Recomendación:** Se recomienda bloquear de inmediato e iniciar contacto de seguridad con el tarjetahabiente. El alto score indica una fuerte desviación estadística y similitud con patrones históricos de fraude.")
            elif tx_row['Probabilidad'] >= 0.20:
                st.warning("⚠️ **Recomendación:** Transacción sospechosa. Poner en cola de revisión manual y auditar las contribuciones de las features para evaluar anomalías en el comportamiento del tarjetahabiente.")
            else:
                st.success("✅ **Recomendación:** Transacción segura. Los parámetros se encuentran dentro de los intervalos normales de uso.")
                
        with col_explain_plot:
            st.markdown("### Contribución SHAP por Feature (Alineación de Causa)")
            st.write("Las características con valores **positivos (Rojo)** incrementan el score de fraude, mientras que las de valores **negativos (Verde)** actúan reduciendo el riesgo.")
            
            # Format feature contributions as dataframe
            explain_df = pd.DataFrame([
                {
                    "feature": f.feature_name,
                    "value": f.value,
                    "contribution": f.contribution
                }
                for f in explanation.features
            ])
            
            # Sort by absolute contribution and take top 12
            explain_df['abs_contribution'] = explain_df['contribution'].abs()
            explain_df = explain_df.sort_values(by='abs_contribution', ascending=False).head(12)
            
            # Color encoding for positive/negative SHAP values
            explain_df['color'] = np.where(explain_df['contribution'] >= 0, '#ef4444', '#10b981')
            
            # Re-order features so larger contributors appear on top of horizontal bar plot
            explain_df = explain_df.sort_values(by='contribution', ascending=True)
            
            fig_shap = go.Figure(go.Bar(
                x=explain_df['contribution'],
                y=explain_df['feature'],
                orientation='h',
                marker_color=explain_df['color'],
                hovertext=[f"Valor original: {v:.4f}" for v in explain_df['value']],
                hoverinfo="x+text"
            ))
            
            fig_shap.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                xaxis_title="Contribución SHAP (Log-Odds)",
                yaxis_title="Feature",
                margin=dict(l=20, r=20, t=10, b=20),
                height=350
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # Highlight Anomaly Score role
            anomaly_feat = [f for f in explanation.features if f.feature_name == 'anomaly_score'][0]
            st.markdown(f"""
                💡 **Aporte de Capa de Anomalía (Fase 1):** 
                La variable de anomalía no supervisada (`anomaly_score`) obtuvo un valor escalado de **`{anomaly_feat.value:.4f}`** 
                y aportó una contribución de **`{anomaly_feat.contribution:+.4f}`** al log-odds del clasificador final.
            """)

# ================= TAB 3: EXPLICABILIDAD GLOBAL =================
with tab3:
    st.markdown("<div class='card-title'>Explicación Global del Modelo (SHAP Feature Importance)</div>", unsafe_allow_html=True)
    st.write("Este gráfico muestra las variables más importantes del modelo general basándose en la contribución absoluta media de SHAP en una muestra representativa del test set.")
    
    # Calculate global explanations for a sample of the dataset
    @st.cache_data
    def get_global_explanations():
        explainer = SHAPExplainer(hybrid_classifier)
        # Sample 100 random rows from pool to calculate mean absolute SHAP values
        sample_indices = np.random.choice(len(X_pool), min(100, len(X_pool)), replace=False)
        X_sample = X_pool[sample_indices]
        
        global_imp = explainer.explain_global(X_sample, feature_names)
        return global_imp
        
    global_importance = get_global_explanations()
    
    # Convert to DataFrame
    global_imp_df = pd.DataFrame(list(global_importance.items()), columns=['Feature', 'SHAP Absoluto Medio'])
    # Take top 15 features
    global_imp_df = global_imp_df.head(15).sort_values(by='SHAP Absoluto Medio', ascending=True)
    
    fig_global = go.Figure(go.Bar(
        x=global_imp_df['SHAP Absoluto Medio'],
        y=global_imp_df['Feature'],
        orientation='h',
        marker_color='#3b82f6'
    ))
    
    fig_global.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        xaxis_title="Importancia SHAP Promedio (Impacto Medio Absoluto)",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=10, b=20),
        height=450
    )
    
    col_global1, col_global2 = st.columns([2, 1])
    with col_global1:
        st.plotly_chart(fig_global, use_container_width=True)
    with col_global2:
        st.markdown("""
            ### Interpretación de Atribución Global
            
            1.  **¿Cuáles características definen el fraude?**
                Las características `V` más arriba en el gráfico representan los factores latentes más informativos para separar transacciones fraudulentas de las normales. Estas representan combinaciones lineales de patrones de comportamiento (localización, tipo de comercio, velocidad, etc.).
                
            2.  **El rol clave del Detector Anómalo (`anomaly_score`):**
                El score de Isolation Forest de la Fase 1 (`anomaly_score`) se ubica entre las variables críticas. Esto demuestra que la detección de comportamientos atípicos no supervisados le da al modelo supervisado XGBoost una robusta ventaja para identificar patrones raros no etiquetados previamente en el entrenamiento.
                
            3.  **Variables `Amount` (Monto) y `Time` (Tiempo):**
                El monto transaccionado y el tiempo cronológico desempeñan un papel catalizador en la decisión del modelo, actuando como contextualizadores de la transacción financiera.
        """)
