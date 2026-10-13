import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Riesgo de ataque al corazón", page_icon="❤️", layout="wide")

VINO = "#7A1F2B"
ROJO = "#C0392B"
ROSA = "#F4D7DB"
GRIS = "#8A8A8A"
CREMA = "#FBF7F6"
TINTA = "#2B1B1E"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: {CREMA}; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {VINO} 0%, #4A1019 100%); }}
section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] .stButton button {{
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.25);
    border-radius: 12px; padding: 0.65rem 1rem; text-align: left; justify-content: flex-start;
    font-weight: 600; transition: all .2s;
}}
section[data-testid="stSidebar"] .stButton button:hover {{ background: rgba(255,255,255,0.22); border-color: #fff; }}
.main .stButton button, div[data-testid="stMainBlockContainer"] .stButton button {{
    background: {VINO}; color: #fff; border: none; border-radius: 12px; font-weight: 600; padding: 0.6rem 1.2rem;
}}
div[data-testid="stMainBlockContainer"] .stButton button:hover {{ background: {ROJO}; color: #fff; }}
div[data-testid="stMainBlockContainer"] .stDownloadButton button {{
    background: #fff; color: {VINO}; border: 2px solid {VINO}; border-radius: 12px; font-weight: 600;
}}
.hero {{
    background: linear-gradient(120deg, {VINO} 0%, {ROJO} 100%); color: #fff;
    padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(122,31,43,0.25);
}}
.hero h1 {{ color: #fff; margin: 0; font-size: 2.3rem; font-weight: 700; }}
.hero p {{ color: #FBE9EB; margin: .4rem 0 0 0; font-size: 1.05rem; }}
.kpi {{
    background: #fff; border-radius: 18px; padding: 1.2rem 1.3rem; border-left: 6px solid {VINO};
    box-shadow: 0 4px 16px rgba(0,0,0,0.06); height: 100%;
}}
.kpi .lbl {{ color: {GRIS}; font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; }}
.kpi .val {{ color: {TINTA}; font-size: 1.9rem; font-weight: 700; line-height: 1.2; }}
.kpi .sub {{ color: {VINO}; font-size: .85rem; }}
.card {{ background: #fff; border-radius: 18px; padding: 1.2rem 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.06); margin-bottom: 1rem; }}
.titulo {{ color: {VINO}; font-weight: 700; font-size: 1.5rem; margin: .4rem 0 .8rem 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {{ background: #FFFFFF; border-color: #FFFFFF; }}
section[data-testid="stSidebar"] .stButton button[kind="primary"] * {{ color: {VINO} !important; }}
div[data-testid="stMainBlockContainer"] .stButton button[kind="secondary"] {{ background: #fff; color: {VINO}; border: 2px solid {VINO}; }}
div[data-testid="stMainBlockContainer"] .stButton button[kind="secondary"]:hover {{ background: {ROSA}; color: {VINO}; }}
div[data-testid="stFormSubmitButton"] button {{ background: {VINO}; color: #fff; border: none; border-radius: 12px; font-weight: 600; padding: .7rem; }}
div[data-testid="stFormSubmitButton"] button:hover {{ background: {ROJO}; color: #fff; }}
.badge-alto {{ background: {ROJO}; color: #fff; padding: .35rem .9rem; border-radius: 999px; font-weight: 600; }}
.badge-bajo {{ background: #2E8B57; color: #fff; padding: .35rem .9rem; border-radius: 999px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

RENOMBRAR = {
    "age": "edad", "hypertension": "hipertension", "heart_disease": "cardiopatia",
    "ever_married": "casado", "avg_glucose_level": "glucosa_promedio",
    "smoking_status": "tabaquismo", "stroke_ataque_corazon": "ataque_corazon",
}
SI_NO = {"Yes": "Sí", "No": "No"}
TABACO = {"never smoked": "Nunca ha fumado", "formerly smoked": "Exfumador",
          "smokes": "Fumador", "Unknown": "Desconocido"}
ORDEN_TABACO = ["Nunca ha fumado", "Exfumador", "Fumador", "Desconocido"]
VARS = ["edad", "hipertension", "cardiopatia", "casado", "glucosa_promedio", "tabaquismo"]


def limpiar(df):
    df = df.rename(columns=RENOMBRAR).copy()
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[c] =df[c].astype(str).str.strip().str.strip("'").str.strip()
    for c in ["hipertension", "cardiopatia", "casado", "ataque_corazon"]:
        if c in df.columns:
            df[c] = df[c].map(SI_NO).fillna(df[c])
    df["tabaquismo"] = df["tabaquismo"].map(TABACO).fillna(df["tabaquismo"])
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["glucosa_promedio"] = pd.to_numeric(df["glucosa_promedio"], errors="coerce")
    return df


@st.cache_data
def cargar(origen):
    datos = pd.read_excel(origen, sheet_name="Datos")
    try:
        futuros = pd.read_excel(origen, sheet_name="Predicciones futuras")
    except ValueError:
        futuros = None
    original = datos.copy()
    df = limpiar(datos)
    n0 = len(df)
    nulos = int(df.isna().sum().sum())
    df = df.dropna()
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    fuera = int(((df["edad"] < 0) | (df["edad"] > 120) | (df["glucosa_promedio"] <= 0)).sum())
    df = df[(df["edad"].between(0, 120)) & (df["glucosa_promedio"] > 0)].reset_index(drop=True)
    df["grupo_edad"] = pd.cut(df["edad"], [0, 18, 30, 45, 60, 75, 120],
                              labels=["0-17", "18-29", "30-44", "45-59", "60-74", "75+"], right=False)
    df["nivel_glucosa"] = pd.cut(df["glucosa_promedio"], [0, 100, 126, 1000],
                                 labels=["Normal (<100)", "Prediabetes (100-125)", "Diabetes (≥126)"], right=False)
    resumen = {"filas_originales": n0, "nulos": nulos, "duplicados": dup, "fuera_rango": fuera, "filas_finales": len(df)}
    if futuros is not None:
        futuros = limpiar(futuros)
    return original, df, futuros, resumen


def a_modelo(df, columnas=None):
    X = df[VARS].copy()
    for c in ["hipertension", "cardiopatia", "casado"]:
        X[c] = (X[c] == "Sí").astype(int)
    X["tabaquismo"] = pd.Categorical(X["tabaquismo"], categories=ORDEN_TABACO)
    X = pd.get_dummies(X, columns=["tabaquismo"], drop_first=True).astype(float)
    if columnas is not None:
        X = X.reindex(columns=columnas, fill_value=0)
    return X


@st.cache_resource
def entrenar(df, nombre):
    X = a_modelo(df)
    y = (df["ataque_corazon"] == "Sí").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    if nombre == "Regresión logística":
        m = LogisticRegression(class_weight="balanced", max_iter=3000)
    else:
        m = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    m.fit(Xtr, ytr)
    prob = m.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    met = {"Exactitud": accuracy_score(yte, pred), "Sensibilidad": recall_score(yte, pred),
           "Precisión": precision_score(yte, pred, zero_division=0), "F1": f1_score(yte, pred),
           "AUC": roc_auc_score(yte, prob)}
    if nombre == "Regresión logística":
        imp = pd.Series(m.coef_[0], index=X.columns)
    else:
        imp = pd.Series(m.feature_importances_, index=X.columns)
    return m, list(X.columns), met, confusion_matrix(yte, pred), roc_curve(yte, prob), imp


def kpi(col, etiqueta, valor, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{etiqueta}</div><div class="val">{valor}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def estilo(fig, alto=380):
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Poppins, sans-serif", color=TINTA), title_font=dict(size=16, color=VINO),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", legend_title_text="")
    return fig


def tasa(df, col):
    t = df.groupby(col, observed=True)["ataque_corazon"].apply(lambda s: (s == "Sí").mean() * 100).reset_index()
    t.columns = [col, "tasa"]
    return t


def medidor(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": " %", "font": {"size": 44, "color": VINO}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": VINO, "thickness": 0.3},
               "steps": [{"range": [0, 30], "color": "#DFF0E5"}, {"range": [30, 60], "color": "#FCEBD2"},
                         {"range": [60, 100], "color": ROSA}],
               "threshold": {"line": {"color": ROJO, "width": 4}, "value": 50}}))
    fig = estilo(fig, 300)
    fig.update_layout(margin=dict(l=40, r=40, t=30, b=10))
    return fig


# ---------------- carga ----------------
ruta = Path(__file__).parent / "ataque_corazon.xlsx"
if ruta.exists():
    origen = str(ruta)
else:
    archivo = st.sidebar.file_uploader("Sube ataque_corazon.xlsx", type=["xlsx"])
    if archivo is None:
        st.info("Sube el archivo ataque_corazon.xlsx en el panel izquierdo para comenzar.")
        st.stop()
    origen = archivo
original, df, futuros, resumen = cargar(origen)

if "pagina" not in st.session_state:
    st.session_state.pagina = "Inicio"
if "modelo" not in st.session_state:
    st.session_state.modelo = "Regresión logística"

with st.sidebar:
    st.markdown("## ❤️ CardioRiesgo")
    st.markdown("Análisis y predicción de ataque al corazón")
    st.markdown("---")
    for icono, nombre in [("🏠", "Inicio"), ("🧹", "Limpieza de datos"), ("📊", "Exploración"),
                          ("🧠", "Modelo"), ("🩺", "Predicción")]:
        if st.button(f"{icono}  {nombre}", key=f"nav_{nombre}", width="stretch",
                     type="primary" if st.session_state.pagina == nombre else "secondary"):
            st.session_state.pagina = nombre
            st.rerun()
    st.markdown("---")
    st.caption(f"Registros limpios: {len(df):,}".replace(",", "."))

pagina = st.session_state.pagina

# ---------------- INICIO ----------------
if pagina == "Inicio":
    st.markdown('<div class="hero"><h1>❤️ Riesgo de ataque al corazón</h1>'
                '<p>Exploración de factores de riesgo y modelo predictivo a partir de datos clínicos de pacientes</p></div>',
                unsafe_allow_html=True)
    casos = int((df["ataque_corazon"] == "Sí").sum())
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Pacientes", f"{len(df):,}".replace(",", "."), "después de limpieza")
    kpi(c2, "Casos de ataque", f"{casos}", f"{casos / len(df) * 100:.1f} % del total")
    kpi(c3, "Edad promedio", f"{df['edad'].mean():.1f}", "años")
    kpi(c4, "Glucosa promedio", f"{df['glucosa_promedio'].mean():.1f}", "mg/dL")
    st.write("")
    a, b = st.columns([1, 1.4])
    with a:
        conteo = df["ataque_corazon"].value_counts().reindex(["No", "Sí"]).reset_index()
        conteo.columns = ["Ataque", "Pacientes"]
        fig = px.pie(conteo, names="Ataque", values="Pacientes", hole=0.62, title="Pacientes con y sin ataque",
                     color="Ataque", color_discrete_map={"No": "#E6D3D6", "Sí": VINO})
        fig.update_traces(textinfo="percent", textfont_size=14)
        st.plotly_chart(estilo(fig), width="stretch")
    with b:
        t = tasa(df, "grupo_edad")
        fig = px.bar(t, x="grupo_edad", y="tasa", title="Tasa de ataque por grupo de edad (%)",
                     text=t["tasa"].map(lambda v: f"{v:.1f}%"), color_discrete_sequence=[VINO])
        fig.update_layout(xaxis_title="Edad", yaxis_title="% con ataque")
        st.plotly_chart(estilo(fig), width="stretch")
    st.markdown("#### Ir a")
    b1, b2, b3, b4 = st.columns(4)
    for col, (txt, dest) in zip([b1, b2, b3, b4], [("🧹 Ver limpieza", "Limpieza de datos"), ("📊 Explorar datos", "Exploración"),
                                                   ("🧠 Entrenar modelo", "Modelo"), ("🩺 Hacer predicción", "Predicción")]):
        if col.button(txt, key=f"ir_{dest}", width="stretch", type="primary"):
            st.session_state.pagina = dest
            st.rerun()

# ---------------- LIMPIEZA ----------------
elif pagina == "Limpieza de datos":
    st.markdown('<div class="titulo">🧹 Limpieza de datos</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Filas originales", f"{resumen['filas_originales']:,}".replace(",", "."))
    kpi(c2, "Valores nulos", resumen["nulos"])
    kpi(c3, "Duplicados eliminados", resumen["duplicados"])
    kpi(c4, "Filas finales", f"{resumen['filas_finales']:,}".replace(",", "."))
    st.write("")
    pasos = pd.DataFrame({
        "Paso": ["Nombres de columnas", "Comillas en tabaquismo", "Valores Yes/No", "Categorías de tabaquismo",
                 "Valores nulos", "Registros duplicados", "Rangos de edad y glucosa", "Variables nuevas"],
        "Acción": ["Se tradujeron al español", "Se quitaron las comillas sobrantes ('never smoked' → never smoked)",
                   "Se cambiaron a Sí / No", "Se tradujeron (Nunca ha fumado, Exfumador, Fumador, Desconocido)",
                   f"Se revisaron: {resumen['nulos']} encontrados", f"Se eliminaron {resumen['duplicados']} filas repetidas",
                   f"Se validaron: {resumen['fuera_rango']} fuera de rango",
                   "Se crearon grupo de edad y nivel de glucosa"]})
    st.dataframe(pasos, hide_index=True, width="stretch")
    t1, t2 = st.tabs(["Datos limpios", "Datos originales"])
    with t1:
        st.dataframe(df.drop(columns=["grupo_edad", "nivel_glucosa"]), height=320, width="stretch")
    with t2:
        st.dataframe(original, height=320, width="stretch")
    st.markdown("#### Estadísticas descriptivas")
    st.dataframe(df[["edad", "glucosa_promedio"]].describe().T.round(2), width="stretch")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.drop(columns=["grupo_edad", "nivel_glucosa"]).to_excel(w, sheet_name="Datos limpios", index=False)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Descargar Excel limpio", buf.getvalue(), "ataque_corazon_limpio.xlsx", width="stretch")
    d2.download_button("⬇️ Descargar CSV limpio", df.drop(columns=["grupo_edad", "nivel_glucosa"]).to_csv(index=False).encode("utf-8-sig"),
                       "ataque_corazon_limpio.csv", "text/csv", width="stretch")

# ---------------- EXPLORACIÓN ----------------
elif pagina == "Exploración":
    st.markdown('<div class="titulo">📊 Exploración de los datos</div>', unsafe_allow_html=True)
    with st.expander("🔎 Filtros", expanded=True):
        f1, f2, f3 = st.columns(3)
        rango = f1.slider("Edad", int(df["edad"].min()), int(df["edad"].max()), (int(df["edad"].min()), int(df["edad"].max())))
        tab = f2.multiselect("Tabaquismo", ORDEN_TABACO, ORDEN_TABACO)
        hip = f3.multiselect("Hipertensión", ["No", "Sí"], ["No", "Sí"])
    d = df[df["edad"].between(*rango) & df["tabaquismo"].isin(tab) & df["hipertension"].isin(hip)]
    if d.empty:
        st.warning("No hay pacientes con esos filtros.")
        st.stop()
    casos = int((d["ataque_corazon"] == "Sí").sum())
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Pacientes filtrados", f"{len(d):,}".replace(",", "."))
    kpi(c2, "Casos de ataque", casos)
    kpi(c3, "Tasa de ataque", f"{casos / len(d) * 100:.2f} %")
    st.write("")
    st.markdown("##### Elige el factor a analizar")
    factores = {"Edad": "grupo_edad", "Hipertensión": "hipertension", "Cardiopatía": "cardiopatia",
                "Tabaquismo": "tabaquismo", "Glucosa": "nivel_glucosa", "Estado civil": "casado"}
    if "factor" not in st.session_state:
        st.session_state.factor = "Edad"
    cols = st.columns(len(factores))
    for col, nombre in zip(cols, factores):
        if col.button(nombre, key=f"fac_{nombre}", width="stretch",
                      type="primary" if st.session_state.factor == nombre else "secondary"):
            st.session_state.factor = nombre
            st.rerun()
    var = factores[st.session_state.factor]
    a, b = st.columns(2)
    with a:
        t = tasa(d, var)
        fig = px.bar(t, x=var, y="tasa", title=f"Tasa de ataque según {st.session_state.factor.lower()} (%)",
                     text=t["tasa"].map(lambda v: f"{v:.1f}%"), color_discrete_sequence=[VINO])
        fig.update_layout(xaxis_title="", yaxis_title="% con ataque")
        st.plotly_chart(estilo(fig), width="stretch")
    with b:
        g = d.groupby([var, "ataque_corazon"], observed=True).size().reset_index(name="Pacientes")
        fig = px.bar(g, x=var, y="Pacientes", color="ataque_corazon", barmode="group",
                     title=f"Pacientes según {st.session_state.factor.lower()}",
                     color_discrete_map={"No": "#E6D3D6", "Sí": VINO})
        fig.update_layout(xaxis_title="")
        st.plotly_chart(estilo(fig), width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.box(d, x="ataque_corazon", y="glucosa_promedio", color="ataque_corazon", points=False,
                     title="Glucosa promedio según ataque", color_discrete_map={"No": "#C9A9AE", "Sí": VINO})
        fig.update_layout(xaxis_title="Ataque al corazón", yaxis_title="mg/dL", showlegend=False)
        st.plotly_chart(estilo(fig), width="stretch")
    with b:
        fig = px.histogram(d, x="edad", color="ataque_corazon", nbins=30, barmode="overlay", opacity=0.75,
                           title="Distribución de edad", color_discrete_map={"No": "#C9A9AE", "Sí": VINO})
        fig.update_layout(xaxis_title="Edad", yaxis_title="Pacientes")
        st.plotly_chart(estilo(fig), width="stretch")
    fig = px.scatter(d.sort_values("ataque_corazon"), x="edad", y="glucosa_promedio", color="ataque_corazon",
                     opacity=0.6, title="Edad vs. glucosa promedio",
                     color_discrete_map={"No": "#D9C2C6", "Sí": VINO})
    fig.update_layout(xaxis_title="Edad", yaxis_title="Glucosa (mg/dL)")
    st.plotly_chart(estilo(fig, 430), width="stretch")

# ---------------- MODELO ----------------
elif pagina == "Modelo":
    st.markdown('<div class="titulo">🧠 Modelo predictivo</div>', unsafe_allow_html=True)
    st.markdown("##### Selecciona el modelo")
    m1, m2, _ = st.columns([1, 1, 2])
    if m1.button("📈 Regresión logística", width="stretch",
                 type="primary" if st.session_state.modelo == "Regresión logística" else "secondary"):
        st.session_state.modelo = "Regresión logística"
        st.rerun()
    if m2.button("🌳 Bosque aleatorio", width="stretch",
                 type="primary" if st.session_state.modelo == "Bosque aleatorio" else "secondary"):
        st.session_state.modelo = "Bosque aleatorio"
        st.rerun()
    nombre = st.session_state.modelo
    modelo, columnas, met, cm, (fpr, tpr, _), imp = entrenar(df, nombre)
    st.markdown(f'<div class="card"><b>Modelo activo:</b> {nombre} · 75 % entrenamiento / 25 % prueba · '
                f'clases balanceadas (solo {(df["ataque_corazon"] == "Sí").mean() * 100:.1f} % de los pacientes tuvo ataque)</div>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    for col, (k, v) in zip(cols, met.items()):
        kpi(col, k, f"{v * 100:.1f} %" if k != "AUC" else f"{v:.3f}")
    st.write("")
    a, b = st.columns(2)
    with a:
        fig = px.imshow(cm, text_auto=True, x=["Pred. No", "Pred. Sí"], y=["Real No", "Real Sí"],
                        color_continuous_scale=["#FFFFFF", ROSA, VINO], title="Matriz de confusión")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textfont_size=18)
        st.plotly_chart(estilo(fig), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=VINO, width=3), fill="tozeroy",
                                 fillcolor="rgba(122,31,43,0.12)", name="Modelo"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dash"), name="Azar"))
        fig.update_layout(title=f"Curva ROC (AUC = {met['AUC']:.3f})", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Sensibilidad")
        st.plotly_chart(estilo(fig), width="stretch")
    imp = imp.rename(lambda s: s.replace("_", " ").replace("tabaquismo ", "Tabaquismo: ").capitalize())
    imp = imp.reindex(imp.abs().sort_values().index)
    titulo = "Coeficientes del modelo" if nombre == "Regresión logística" else "Importancia de las variables"
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", title=titulo,
                 color=imp.values > 0, color_discrete_map={True: VINO, False: "#C9A9AE"})
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(estilo(fig, 360), width="stretch")

# ---------------- PREDICCIÓN ----------------
elif pagina == "Predicción":
    st.markdown('<div class="titulo">🩺 Predicción de riesgo</div>', unsafe_allow_html=True)
    nombre = st.session_state.modelo
    modelo, columnas, *_ = entrenar(df, nombre)
    st.caption(f"Modelo en uso: {nombre} (se cambia en la sección Modelo)")
    t1, t2 = st.tabs(["👤 Paciente nuevo", "📋 Predicciones futuras del archivo"])
    with t1:
        with st.form("paciente"):
            a, b, c = st.columns(3)
            edad = a.number_input("Edad", 0, 110, 55)
            glucosa = a.number_input("Glucosa promedio (mg/dL)", 40.0, 300.0, 100.0, step=0.5)
            hip = b.radio("Hipertensión", ["No", "Sí"], horizontal=True)
            car = b.radio("Cardiopatía", ["No", "Sí"], horizontal=True)
            cas = c.radio("¿Ha estado casado?", ["No", "Sí"], index=1, horizontal=True)
            tab = c.selectbox("Tabaquismo", ORDEN_TABACO)
            enviar = st.form_submit_button("❤️ Calcular riesgo", width="stretch")
        if enviar:
            p = pd.DataFrame([{"edad": edad, "hipertension": hip, "cardiopatia": car, "casado": cas,
                               "glucosa_promedio": glucosa, "tabaquismo": tab}])
            prob = modelo.predict_proba(a_modelo(p, columnas))[0, 1]
            x, y = st.columns([1.2, 1])
            with x:
                st.plotly_chart(medidor(prob), width="stretch")
            with y:
                st.write("")
                st.write("")
                if prob >= 0.5:
                    st.markdown('<span class="badge-alto">Riesgo ALTO</span>', unsafe_allow_html=True)
                    st.markdown("El perfil del paciente se parece al de quienes sufrieron un ataque en los datos.")
                else:
                    st.markdown('<span class="badge-bajo">Riesgo BAJO</span>', unsafe_allow_html=True)
                    st.markdown("El perfil del paciente se parece al de quienes no sufrieron un ataque en los datos.")
                st.caption("Resultado académico; no reemplaza una valoración médica.")
    with t2:
        if futuros is None:
            st.info("El archivo no tiene la hoja 'Predicciones futuras'.")
        else:
            st.dataframe(futuros, hide_index=True, width="stretch")
            if st.button("🔮 Predecir pacientes futuros", type="primary"):
                r = futuros.copy()
                r["probabilidad"] = modelo.predict_proba(a_modelo(r, columnas))[:, 1]
                r["predicción"] = np.where(r["probabilidad"] >= 0.5, "Sí", "No")
                r.insert(0, "paciente", [f"Paciente {i + 1}" for i in range(len(r))])
                fig = px.bar(r, x="paciente", y="probabilidad", text=r["probabilidad"].map(lambda v: f"{v * 100:.1f}%"),
                             color="predicción", color_discrete_map={"Sí": VINO, "No": "#C9A9AE"},
                             title="Probabilidad estimada de ataque")
                fig.add_hline(y=0.5, line_dash="dash", line_color=ROJO)
                fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_range=[0, 1])
                st.plotly_chart(estilo(fig), width="stretch")
                r["probabilidad"] = r["probabilidad"].map(lambda v: f"{v * 100:.1f} %")
                st.dataframe(r, hide_index=True, width="stretch")
