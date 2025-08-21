"""
Requisitos (requirements.txt):
--------------------------------
streamlit
pandas
sqlalchemy
psycopg2-binary
plotly
python-dateutil

Segredos (.streamlit/secrets.toml):
-----------------------------------
[db]
# Ex.: Supabase/Postgres
url = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
# Para testes locais sem segredos, o app cai automaticamente para SQLite (dados.db)
"""

import io
import uuid
from datetime import datetime, date
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dateutil import parser as dateparser

# ---------------------------
# Config inicial
# ---------------------------
st.set_page_config(page_title="Reajuste Convênios Oficial", layout="wide")

REGIONS = ["AL","BA","PA","PE","RJ","DF","SC","SP"]
UI_COLS = [
    "REGIONAL","MARCA","OPERADORA","TIPO","DATA DO CONTRATO","DATA ULTIMO REAJUSTE",
    "DATA PREVISTA","STATUS","RESPONSÁVEL","% PREVISTO","DATA REALIZADA","% REALIZADO",
    "CONSULTAS","EXAMES","CIRURGIAS","HM","TAXA","OBS","ROB PREVISTA","ROB REALIZADA","ROB REAJUSTE"
]
DATE_COLS = ["DATA DO CONTRATO","DATA ULTIMO REAJUSTE","DATA PREVISTA","DATA REALIZADA"]
INT_COLS = ["CONSULTAS","EXAMES","CIRURGIAS","HM"]
FLOAT_COLS = ["% PREVISTO","% REALIZADO","TAXA","ROB PREVISTA","ROB REALIZADA","ROB REAJUSTE"]

# Mapear UI -> DB (snake_case)
COLMAP: Dict[str, str] = {
    "id": "id",
    "REGIONAL": "regional",
    "MARCA": "marca",
    "OPERADORA": "operadora",
    "TIPO": "tipo",
    "DATA DO CONTRATO": "data_do_contrato",
    "DATA ULTIMO REAJUSTE": "data_ultimo_reajuste",
    "DATA PREVISTA": "data_prevista",
    "STATUS": "status",
    "RESPONSÁVEL": "responsavel",
    "% PREVISTO": "pct_previsto",
    "DATA REALIZADA": "data_realizada",
    "% REALIZADO": "pct_realizado",
    "CONSULTAS": "consultas",
    "EXAMES": "exames",
    "CIRURGIAS": "cirurgias",
    "HM": "hm",
    "TAXA": "taxa",
    "OBS": "obs",
    "ROB PREVISTA": "rob_prevista",
    "ROB REALIZADA": "rob_realizada",
    "ROB REAJUSTE": "rob_reajuste",
}
DB_COLS: List[str] = [COLMAP[c] for c in UI_COLS]

# ---------------------------
# Conexão ao banco
# ---------------------------
# substitua a função get_engine() por esta
import os

@st.cache_resource(show_spinner=False)
def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # fallback: secrets.toml local
        try:
            db_url = st.secrets["db"]["url"]
        except Exception:
            db_url = None
    if not db_url:
        db_url = "sqlite:///dados.db"  # último fallback p/ testes
        st.sidebar.info("Sem DATABASE_URL/st.secrets -> usando SQLite local (dados.db)")
    # Render e outros às vezes entregam 'postgres://', troque o prefixo:
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://")
    return create_engine(db_url, pool_pre_ping=True)
    return engine

engine = get_engine()

# ---------------------------
# DDL (create table if not exists)
# ---------------------------
DDL = """
CREATE TABLE IF NOT EXISTS contratos (
  id TEXT PRIMARY KEY,
  regional TEXT NOT NULL,
  marca TEXT,
  operadora TEXT,
  tipo TEXT,
  data_do_contrato DATE,
  data_ultimo_reajuste DATE,
  data_prevista DATE,
  status TEXT,
  responsavel TEXT,
  pct_previsto NUMERIC,
  data_realizada DATE,
  pct_realizado NUMERIC,
  consultas INTEGER,
  exames INTEGER,
  cirurgias INTEGER,
  hm INTEGER,
  taxa NUMERIC,
  obs TEXT,
  rob_prevista NUMERIC,
  rob_realizada NUMERIC,
  rob_reajuste NUMERIC,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
with engine.begin() as conn:
    conn.execute(text(DDL))

# Trigger para updated_at (Postgres). Ignorado silenciosamente em SQLite.
TRIGGER_FN = """
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""
TRIGGER = """
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'set_updated_at') THEN
    -- ok
  ELSE
    PERFORM 1; -- noop
  END IF;
END $$;
CREATE TRIGGER contratos_set_updated_at
BEFORE UPDATE ON contratos
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
"""
try:
    with engine.begin() as conn:
        conn.execute(text(TRIGGER_FN))
        conn.execute(text(TRIGGER))
except Exception:
    # Provavelmente não é Postgres; tudo bem
    pass

# ---------------------------
# Helpers de tipagem e normalização
# ---------------------------

def to_date_safe(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (datetime, date)):
        return v if isinstance(v, date) else v.date()
    try:
        return dateparser.parse(str(v)).date()
    except Exception:
        return None


def normalize_df_ui(df: pd.DataFrame) -> pd.DataFrame:
    # Garante colunas na ordem e presentes
    out = df.copy()
    for c in ["id"] + UI_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[["id"] + UI_COLS]

    # Tipos na UI
    for c in DATE_COLS:
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
    for c in INT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in FLOAT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Preenche REGIONAL vazio com primeira regional se houver
    out["REGIONAL"] = out["REGIONAL"].astype(object)
    return out


def ui_to_db_rows(df_ui: pd.DataFrame) -> List[Dict]:
    rows = []
    for _, row in df_ui.iterrows():
        r = {}
        r["id"] = row.get("id") or str(uuid.uuid4())
        for uicol, dbcol in COLMAP.items():
            if uicol == "id":
                continue
            val = row.get(uicol)
            if uicol in DATE_COLS:
                val = to_date_safe(val)
            elif uicol in INT_COLS:
                val = None if pd.isna(val) else int(val)
            elif uicol in FLOAT_COLS:
                val = None if pd.isna(val) else float(val)
            else:
                if pd.isna(val) if isinstance(val, float) else (val is None):
                    val = None
            r[dbcol] = val
        rows.append(r)
    return rows

# ---------------------------
# CRUD
# ---------------------------
@st.cache_data(ttl=30)
def fetch_region(region: str) -> pd.DataFrame:
    q = text("SELECT id, " + ", ".join(DB_COLS) + " FROM contratos WHERE regional = :r ORDER BY updated_at DESC, created_at DESC")
    with engine.begin() as conn:
        res = conn.execute(q, {"r": region}).mappings().all()
    if not res:
        df = pd.DataFrame(columns=["id"] + UI_COLS)
    else:
        df = pd.DataFrame(res)
        # Converter nomes de colunas DB -> UI
        rename = {v: k for k, v in COLMAP.items() if k != "id"}
        df = df.rename(columns=rename)
        df = df[["id"] + UI_COLS]
    return normalize_df_ui(df)

@st.cache_data(ttl=30)
def fetch_all() -> pd.DataFrame:
    q = text("SELECT id, " + ", ".join(DB_COLS) + " FROM contratos")
    with engine.begin() as conn:
        res = conn.execute(q).mappings().all()
    if not res:
        df = pd.DataFrame(columns=["id"] + UI_COLS)
    else:
        df = pd.DataFrame(res)
        rename = {v: k for k, v in COLMAP.items() if k != "id"}
        df = df.rename(columns=rename)
        df = df[["id"] + UI_COLS]
    return normalize_df_ui(df)


def upsert_rows(rows: List[Dict]):
    if not rows:
        return
    cols_db = ["id"] + DB_COLS
    placeholders = ", ".join([":" + c for c in cols_db])
    insert_sql = text(f"INSERT INTO contratos (" + ", ".join(cols_db) + ") VALUES (" + placeholders + ")")
    # Simples: tentamos update, se não atualizou, fazemos insert
    with engine.begin() as conn:
        for r in rows:
            # UPDATE
            set_part = ", ".join([f"{c} = :{c}" for c in DB_COLS])
            upd = text(f"UPDATE contratos SET {set_part} WHERE id = :id")
            result = conn.execute(upd, r)
            if result.rowcount == 0:
                conn.execute(insert_sql, r)


def delete_missing(region: str, keep_ids: List[str]):
    if not keep_ids:
        return
    q = text("DELETE FROM contratos WHERE regional = :r AND id NOT IN (" + ",".join([":id"+str(i) for i in range(len(keep_ids))]) + ")")
    params = {"r": region}
    params.update({"id"+str(i): keep_ids[i] for i in range(len(keep_ids))})
    with engine.begin() as conn:
        conn.execute(q, params)


def sync_region(region: str, df_ui: pd.DataFrame):
    # Garante colunas e id
    df = normalize_df_ui(df_ui)
    df["REGIONAL"] = region
    # Gera ids onde faltar
    if "id" not in df.columns:
        df.insert(0, "id", None)
    df["id"] = df["id"].apply(lambda x: x if (isinstance(x, str) and len(x) > 5) else str(uuid.uuid4()))

    rows = ui_to_db_rows(df)
    upsert_rows(rows)
    # Deletar os que sumiram da grade
    keep_ids = df["id"].tolist()
    delete_missing(region, keep_ids)

    fetch_region.clear()  # limpa cache
    fetch_all.clear()

# ---------------------------
# UI
# ---------------------------
st.sidebar.title("Painel – Reajuste Convênios 2025")
page = st.sidebar.radio("Navegação", ["Dashboard", "Regionais (editar)", "Consolidado (visualização)", "Importar inicial (Excel→Banco)"])

# DASHBOARD
if page == "Dashboard":
    st.header("Dashboard")
    df_all = fetch_all()

    with st.expander("Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        regions_sel = c1.multiselect("Regionais", REGIONS, default=REGIONS)
        status_sel = c2.multiselect("Status", sorted([x for x in df_all["STATUS"].dropna().unique().tolist()]))
        oper_sel = c3.multiselect("Operadora", sorted([x for x in df_all["OPERADORA"].dropna().unique().tolist()]))

    mask = df_all["REGIONAL"].isin(regions_sel)
    if status_sel:
        mask &= df_all["STATUS"].isin(status_sel)
    if oper_sel:
        mask &= df_all["OPERADORA"].isin(oper_sel)
    dff = df_all[mask].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total registros", len(dff))
    c2.metric("% previsto (média)", f"{pd.to_numeric(dff['% PREVISTO'], errors='coerce').mean():.1f}%")
    c3.metric("% realizado (média)", f"{pd.to_numeric(dff['% REALIZADO'], errors='coerce').mean():.1f}%")
    c4.metric("Consultas (soma)", int(pd.to_numeric(dff['CONSULTAS'], errors='coerce').fillna(0).sum()))

    if len(dff):
        g1 = (dff.groupby("REGIONAL", as_index=False)
                .agg({"% PREVISTO":"mean","% REALIZADO":"mean"}))
        fig1 = px.bar(g1, x="REGIONAL", y=["% PREVISTO","% REALIZADO"], barmode="group", title="% Previsto vs Realizado por Regional")
        st.plotly_chart(fig1, use_container_width=True)

        g2 = (dff.groupby("OPERADORA", as_index=False)
                .agg({"CONSULTAS":"sum","EXAMES":"sum","CIRURGIAS":"sum","HM":"sum"}))
        fig2 = px.bar(g2, x="OPERADORA", y=["CONSULTAS","EXAMES","CIRURGIAS","HM"], title="Volume por Operadora")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabela filtrada")
    st.dataframe(dff, use_container_width=True, height=480)

# REGIONAIS – EDIÇÃO
elif page == "Regionais (editar)":
    st.header("Edição por Regional (CRUD)")
    reg = st.selectbox("Escolha a regional", REGIONS, index=0)
    df = fetch_region(reg)

    st.caption("Edite os dados abaixo. Use o botão **Adicionar linha** para novos registros.")
    # Adiciona linha em branco
    if st.button("➕ Adicionar linha"):
        blank = {c: None for c in ["id"] + UI_COLS}
        blank["REGIONAL"] = reg
        df = pd.concat([pd.DataFrame([blank]), df], ignore_index=True)

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_order=["id"] + UI_COLS,
        disabled=["id","REGIONAL"],  # REGIONAL fixada pela aba selecionada
        column_config={
            "id": st.column_config.TextColumn(help="Identificador do registro (não editável)"),
            "% PREVISTO": st.column_config.NumberColumn(format="%.2f"),
            "% REALIZADO": st.column_config.NumberColumn(format="%.2f"),
            "TAXA": st.column_config.NumberColumn(format="%.6f"),
            "DATA DO CONTRATO": st.column_config.DateColumn(),
            "DATA ULTIMO REAJUSTE": st.column_config.DateColumn(),
            "DATA PREVISTA": st.column_config.DateColumn(),
            "DATA REALIZADA": st.column_config.DateColumn(),
        },
        hide_index=True,
    )

    c1, c2 = st.columns([1,1])
    if c1.button("💾 Salvar alterações na regional", type="primary"):
        try:
            sync_region(reg, edited)
            st.success(f"{reg}: alterações salvas.")
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")

    if c2.button("↩️ Recarregar da base"):
        fetch_region.clear()
        st.experimental_rerun()

# CONSOLIDADO – VISUALIZAÇÃO
else:
    st.header("Consolidado (somente leitura)")
    df_all = fetch_all()
    st.dataframe(df_all, use_container_width=True, height=520)

    if len(df_all):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df_all.to_excel(xw, index=False, sheet_name="Consolidado")
        st.download_button("⬇️ Baixar consolidado (.xlsx)", data=buf.getvalue(),
                           file_name=f"Consolidado_{date.today().isoformat()}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------------
# IMPORTAR INICIAL – EXCEL → BANCO (uma vez)
# -----------------------------
elif page == "Importar inicial (Excel→Banco)":
    st.header("Importar inicial a partir do Excel (uma vez)")
    st.caption("Carregue o Excel atual com as abas AL, BA, PA, PE, RJ, DF, SC, SP. Após importar, a edição será **somente pelo app**.")

    def count_rows() -> int:
        with engine.begin() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM contratos")).scalar() or 0

    def delete_all():
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM contratos"))

    up = st.file_uploader("Envie o arquivo .xlsx", type=["xlsx"]) 
    current_rows = count_rows()
    st.info(f"Registros atuais na base: {current_rows}")

    mode = st.radio("Modo de importação", ["Anexar (não apaga dados existentes)", "Substituir (apaga e reimporta)"])
    confirm = st.checkbox("Confirmo que esta importação é intencional")

    def read_excel_file(file) -> pd.DataFrame:
        xls = pd.ExcelFile(file)
        dfs = []
        for reg in REGIONS:
            if reg not in xls.sheet_names:
                st.warning(f"Aba {reg} não encontrada; será ignorada.")
                continue
            dfr = pd.read_excel(xls, sheet_name=reg, dtype=object)
            # Normaliza colunas/ordem
            for c in UI_COLS:
                if c not in dfr.columns:
                    dfr[c] = None
            dfr = dfr[UI_COLS].copy()
            # Tipos e saneamento
            for c in DATE_COLS:
                dfr[c] = pd.to_datetime(dfr[c], errors="coerce").dt.date
            for c in INT_COLS:
                dfr[c] = pd.to_numeric(dfr[c], errors="coerce").astype("Int64")
            for c in FLOAT_COLS:
                dfr[c] = pd.to_numeric(dfr[c], errors="coerce")
            # Garante REGIONAL
            dfr["REGIONAL"] = dfr["REGIONAL"].fillna(reg)
            # Remove linhas completamente vazias
            mask_any = dfr.drop(columns=["OBS"]).notna().any(axis=1)
            dfr = dfr[mask_any]
            dfs.append(dfr)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=UI_COLS)

    if up is not None:
        with st.spinner("Lendo arquivo..."):
            df_import = read_excel_file(up)
        st.subheader("Prévia dos dados a importar")
        st.dataframe(df_import.head(100), use_container_width=True, height=400)
        st.write(f"Total a importar: **{len(df_import)}** linhas")

        if st.button("🚀 Executar importação") and confirm:
            try:
                if mode.startswith("Substituir"):
                    delete_all()
                # Gera ids (para novos) e envia em lote por regional
                df_imp = df_import.copy()
                if "id" not in df_imp.columns:
                    df_imp.insert(0, "id", None)
                df_imp["id"] = df_imp["id"].apply(lambda x: x if (isinstance(x, str) and len(x) > 5) else str(uuid.uuid4()))

                # Upsert em blocos por regional
                for reg in REGIONS:
                    block = df_imp[df_imp["REGIONAL"].astype(str) == reg]
                    if len(block):
                        sync_region(reg, block)
                fetch_all.clear(); fetch_region.clear()
                st.success("Importação concluída. A partir de agora, edite somente pelo app.")
            except Exception as e:
                st.error(f"Falha na importação: {e}")
        elif st.button("🚫 Cancelar operação"):
            st.stop()", data=buf.getvalue(),
                           file_name=f"Consolidado_{date.today().isoformat()}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
