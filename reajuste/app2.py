"""
Requisitos (requirements.txt)
-----------------------------
streamlit
pandas
plotly
python-dateutil
openpyxl
"""

import os
import io
import re
import uuid
import unicodedata
from datetime import datetime, date
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px  # não usado no dashboard atual; pode remover se quise
from dateutil import parser as dateparser

# ---------------------------
# Config inicial
# ---------------------------
st.set_page_config(page_title="Reajuste Convênios 2025", layout="wide")

# Diretório de dados com fallback (para Render Free sem Disk)
DEFAULT_WRITABLE = os.path.join(os.path.dirname(__file__), "data")
FALLBACKS = [
    os.getenv("DATA_DIR", "/var/data"),  # tenta Disk, se existir
    DEFAULT_WRITABLE,                    # pasta do app (efêmero em cloud)
    "/tmp/data",                         # sempre gravável (efêmero)
]
for candidate in FALLBACKS:
    try:
        os.makedirs(candidate, exist_ok=True)
        DATA_DIR = candidate
        break
    except PermissionError:
        continue

CSV_PATH = os.path.join(DATA_DIR, "dados.csv")

# (opcional) semear dados a partir de um CSV comitado no repo
SEED_CSV = os.path.join(os.path.dirname(__file__), "seed_dados.csv")
if (not os.path.exists(CSV_PATH)) and os.path.exists(SEED_CSV):
    import shutil
    shutil.copy2(SEED_CSV, CSV_PATH)

REGIONS = ["AL", "BA", "PA", "PE", "RJ", "DF", "SC", "SP"]

# Nome canônico e sinônimo aceito na importação
NEW_COL_CANON = "PARAMETRIZAÇÃO"
IMPORT_SYNONYMS = {"PARAMETRIZACAO": NEW_COL_CANON}

# Colunas da UI (ordem de exibição)
UI_COLS = [
    "REGIONAL", "MARCA", "OPERADORA", "TIPO",
    "DATA DO CONTRATO", "DATA ULTIMO REAJUSTE", "DATA PREVISTA",
    "STATUS", "RESPONSÁVEL", "% PREVISTO", "DATA REALIZADA", "% REALIZADO",
    "CONSULTAS", "EXAMES", "CIRURGIAS", "HM", "TAXA", "OBS",
    "ROB PREVISTA", "ROB REALIZADA", "ROB REAJUSTE",
    NEW_COL_CANON,  # PARAMETRIZAÇÃO (data)
]

# Tipos por coluna (padrão final)
TEXT_COLS = ["REGIONAL", "MARCA", "OPERADORA", "TIPO", "STATUS", "RESPONSÁVEL", "OBS"]

DATE_COLS = [
    "DATA DO CONTRATO", "DATA ULTIMO REAJUSTE",
    "DATA PREVISTA", "DATA REALIZADA",
    NEW_COL_CANON,  # PARAMETRIZAÇÃO
]

# Percentuais (0–100)
PCT_COLS = ["% PREVISTO", "% REALIZADO", "TAXA", "CONSULTAS", "EXAMES", "CIRURGIAS", "HM"]

# Inteiros (nenhum, pois as 4 categorias são %)
INT_COLS: List[str] = []

# Decimais / moeda
DEC_COLS = ["ROB PREVISTA", "ROB REALIZADA", "ROB REAJUSTE"]

# Mapa UI->“DB” – no modo CSV mantemos os nomes da UI
COLMAP: Dict[str, str] = {c: c for c in UI_COLS}
COLMAP["id"] = "id"

# ---------------------------
# Conversores e normalização
# ---------------------------
def to_date_series(col):
    s = pd.Series(col)
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    nums = pd.to_numeric(s, errors="coerce")
    num_mask = nums.notna()
    if num_mask.any():
        arr = nums[num_mask].astype("float64").to_numpy()
        dt_num = pd.to_datetime(arr, unit="D", origin="1899-12-30", errors="coerce")
        dt.loc[num_mask] = dt_num

    rem_mask = dt.isna()
    if rem_mask.any():
        def _parse_one(x):
            if pd.isna(x): return pd.NaT
            if isinstance(x, (datetime, date)): return pd.Timestamp(x)
            try:
                return pd.Timestamp(dateparser.parse(str(x), dayfirst=True))
            except Exception:
                return pd.NaT
        dt.loc[rem_mask] = s[rem_mask].apply(_parse_one)

    return dt.dt.date

def to_date_safe(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, (datetime, date)):
        return v if isinstance(v, date) else v.date()
    try:
        return dateparser.parse(str(v), dayfirst=True).date()
    except Exception:
        return None

def to_percent_0_100(s) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or (isinstance(s, str) and not s.strip()):
        return None
    txt = re.sub(r"[^0-9,.\-%]", "", str(s)).strip()
    has_pct = "%" in txt
    txt = txt.replace("%", "")
    if re.search(r"\d,\d+$", txt):  # vírgula decimal BR
        txt = txt.replace(".", "").replace(",", ".")
    try:
        v = float(txt)
    except Exception:
        return None
    if not has_pct and v <= 1:
        v = v * 100.0
    if v < -1000 or v > 1000:
        return None
    if v < 0: v = 0.0
    if v > 100: v = 100.0
    return v

def to_decimal(s) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or (isinstance(s, str) and not s.strip()):
        return None
    txt = re.sub(r"[Rr]\$|\s", "", str(s))
    if re.search(r"\d,\d{1,2}$", txt):
        txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except Exception:
        return None

def normalize_df_ui(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["id"] + UI_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[["id"] + UI_COLS]

    # Datas
    for c in DATE_COLS:
        out[c] = to_date_series(out[c])

    # Inteiros (se houvesse)
    for c in INT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")

    # Percentuais
    for c in PCT_COLS:
        out[c] = out[c].apply(to_percent_0_100)

    # Decimais
    for c in DEC_COLS:
        out[c] = out[c].apply(to_decimal)

    # Texto
    for c in TEXT_COLS:
        out[c] = out[c].astype(str).str.strip().replace({"": None, "nan": None, "None": None})

    return out

# ---------------------------
# Armazenamento em CSV
# ---------------------------
def csv_init_if_needed():
    if not os.path.exists(CSV_PATH):
        empty = pd.DataFrame(columns=["id"] + UI_COLS)
        empty.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

def csv_load() -> pd.DataFrame:
    csv_init_if_needed()
    try:
        df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, na_values=[""])
    except Exception:
        df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, na_values=[""], encoding="latin-1")
    return normalize_df_ui(df)

def csv_save(df_ui: pd.DataFrame):
    df = normalize_df_ui(df_ui).copy()
    if "id" not in df.columns:
        df.insert(0, "id", None)
    df["id"] = df["id"].apply(lambda x: x if (isinstance(x, str) and len(x) > 5) else str(uuid.uuid4()))

    df_to = df.copy()
    # Datas ISO no arquivo
    for c in DATE_COLS:
        df_to[c] = pd.to_datetime(df_to[c], errors="coerce").dt.strftime("%Y-%m-%d")
        df_to.loc[df_to[c].isna(), c] = ""
    # Percentuais/decimais como string
    for c in PCT_COLS + DEC_COLS:
        df_to[c] = df_to[c].apply(lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x)}")
    # Textos
    for c in TEXT_COLS:
        df_to[c] = df_to[c].apply(lambda x: "" if x in (None, "None", "nan") else str(x))

    df_to.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# ---------------------------
# Fallbacks seguros de salvamento (tudo texto)
# ---------------------------
def _df_ui_region_to_text(df_ui: pd.DataFrame, region: str) -> pd.DataFrame:
    """Normaliza só o essencial e converte TUDO para texto seguro para persistir."""
    df = df_ui.copy()
    for c in ["id"] + UI_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[["id"] + UI_COLS].copy()

    df["REGIONAL"] = str(region).strip().upper()
    df["id"] = df["id"].apply(lambda x: x if (isinstance(x, str) and len(x) > 5) else str(uuid.uuid4()))

    for c in DATE_COLS:
        df[c] = df[c].apply(lambda v: (to_date_safe(v).isoformat() if to_date_safe(v) else ""))

    for c in [col for col in UI_COLS if col not in DATE_COLS]:
        df[c] = df[c].apply(lambda v: "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))

    return df

def _save_region_all_text(region: str, df_ui: pd.DataFrame):
    """Sobrescreve a regional no CSV usando somente strings (sem casts numéricos)."""
    df_all = csv_load()
    block_txt = _df_ui_region_to_text(df_ui, region)
    rest = df_all[df_all["REGIONAL"] != region].copy()

    for c in DATE_COLS:
        rest[c] = rest[c].apply(lambda v: (to_date_safe(v).isoformat() if to_date_safe(v) else ""))
    for c in [col for col in UI_COLS if col not in DATE_COLS]:
        rest[c] = rest[c].apply(lambda v: "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))

    merged = pd.concat([rest[["id"] + UI_COLS], block_txt[["id"] + UI_COLS]], ignore_index=True)
    merged.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# ---------------------------
# Cache/CRUD em modo arquivo
# ---------------------------
@st.cache_data(ttl=30)
def fetch_region(region: str) -> pd.DataFrame:
    df_all = csv_load()
    return df_all[df_all["REGIONAL"] == region].copy()

@st.cache_data(ttl=30)
def fetch_all() -> pd.DataFrame:
    return csv_load()

def sync_region(region: str, df_ui: pd.DataFrame):
    """
    1) Tenta o caminho normal (com normalização completa).
    2) Se der qualquer erro, aplica fallback all-text.
    """
    try:
        df_all = csv_load()
        df_new = normalize_df_ui(df_ui).copy()
        df_new["REGIONAL"] = region
        if "id" not in df_new.columns:
            df_new.insert(0, "id", None)
        df_new["id"] = df_new["id"].apply(lambda x: x if (isinstance(x, str) and len(x) > 5) else str(uuid.uuid4()))
        df_rest = df_all[df_all["REGIONAL"] != region].copy()
        merged = pd.concat([df_rest, df_new], ignore_index=True)
        csv_save(merged)
    except Exception as e:
        st.warning(f"Salvamento normal falhou ({e}). Aplicando modo seguro (texto).")
        _save_region_all_text(region, df_ui)

    fetch_region.clear()
    fetch_all.clear()

# ---------------------------
# Helpers do Dashboard (formatação e estilo)
# ---------------------------
DAYS_THRESHOLD = 90  # dias para alerta de vencimento

def strip_accents(txt: str) -> str:
    if txt is None:
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", str(txt)) if unicodedata.category(c) != "Mn")

def row_highlight_style(row):
    """
    Destaque por linha:
      - STATUS = EM ATRASO -> vermelho claro
      - "no prazo" (ou rótulos equivalentes) com DATA PREVISTA <= 90 dias -> amarelo
    """
    status = str(row.get("STATUS", "")).strip().upper()
    dt_prev = pd.to_datetime(row.get("DATA PREVISTA", None), errors="coerce", dayfirst=True)
    today = pd.Timestamp(date.today())
    soon = pd.notna(dt_prev) and 0 <= (dt_prev - today).days <= DAYS_THRESHOLD

    if status == "EM ATRASO":
        return ['background-color: #FFECEC; border-left: 4px solid #D00000;'] * len(row)

    is_no_prazo_label = status in {"NO PRAZO", "EM DIA", "EM PRAZO", "DENTRO DO PRAZO"}
    is_other_ok = status not in {"REALIZADO", "EM ATRASO", ""}
    if (is_no_prazo_label or is_other_ok) and soon:
        return ['background-color: #FFF4CC; border-left: 4px solid #E6A700;'] * len(row)

    return [''] * len(row)

def format_df_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formata datas (pt-BR) e percentuais (3,5%) apenas para exibição no Dashboard.
    """
    out = df.copy()

    # Datas -> dd/mm/yyyy
    for c in DATE_COLS:
        s = pd.to_datetime(out[c], errors="coerce")
        out[c] = s.dt.strftime("%d/%m/%Y")
        out.loc[s.isna(), c] = ""

    # Percentuais -> '3,5%'
    for c in PCT_COLS:
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.map(lambda x: "" if pd.isna(x) else f"{x:.1f}".replace(".", ",") + "%")

    return out

# ---------------------------
# UI
# ---------------------------
st.sidebar.title("Painel – Reajuste Convênios 2025")
st.sidebar.caption(f"Armazenamento: {os.path.abspath(CSV_PATH)}")
page = st.sidebar.radio(
    "Navegação",
    ["Dashboard", "Regionais (editar)", "Importar inicial (Excel→Arquivo)"]
)

# DASHBOARD
if page == "Dashboard":
    st.header("Dashboard")
    df_all = fetch_all()
    st.caption(f"Registros carregados: **{len(df_all)}**")

    with st.expander("Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        regions_sel = c1.multiselect("Regionais", REGIONS, default=REGIONS)
        status_sel  = c2.multiselect("Status", sorted([x for x in df_all["STATUS"].dropna().unique().tolist()]))
        oper_sel    = c3.multiselect("Operadora", sorted([x for x in df_all["OPERADORA"].dropna().unique().tolist()]))

    mask = df_all["REGIONAL"].isin(regions_sel)
    if status_sel:
        mask &= df_all["STATUS"].isin(status_sel)
    if oper_sel:
        mask &= df_all["OPERADORA"].isin(oper_sel)
    dff = df_all[mask].copy()

    # -------- Indicadores (linha 1) --------
    s = dff["STATUS"].astype(str).str.strip().str.upper().fillna("")
    total = len(dff)
    qtd_realizado = int((s == "REALIZADO").sum())
    qtd_atraso    = int((s == "EM ATRASO").sum())

    # "No prazo": rótulos explícitos; se não houver, usa o restante
    no_prazo_labels = {"NO PRAZO", "EM DIA", "EM PRAZO", "DENTRO DO PRAZO"}
    mask_no_prazo_labels = s.isin(no_prazo_labels)
    if int(mask_no_prazo_labels.sum()) == 0:
        mask_no_prazo = ~s.isin({"REALIZADO", "EM ATRASO", ""})
    else:
        mask_no_prazo = mask_no_prazo_labels
    qtd_no_prazo = int(mask_no_prazo.sum())

    # Converte uma única vez para usar em “em negociação” e depois reusar
    dt_prev = pd.to_datetime(dff["DATA PREVISTA"], errors="coerce", dayfirst=True)
    today = pd.Timestamp(date.today())
    delta_days = (dt_prev - today).dt.days
    mask_soon = dt_prev.notna() & (delta_days >= 0) & (delta_days <= DAYS_THRESHOLD)
    qtd_em_negociacao = int((mask_no_prazo & mask_soon).sum())

    # 5 indicadores em uma linha
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", total)
    c2.metric("Realizado", qtd_realizado)
    c3.metric("Em atraso", qtd_atraso)
    c4.metric("No prazo", qtd_no_prazo)
    c5.metric("Em negociação", qtd_em_negociacao)

    # -------- Indicadores (linha 2: TIPO) --------
    tipo_norm = dff["TIPO"].astype(str).map(strip_accents).str.upper().str.strip()
    qtd_conv   = int((tipo_norm == "CONVENIO").sum())
    qtd_public = int((tipo_norm == "PUBLICO").sum())
    qtd_sus    = int((tipo_norm == "SUS").sum())

    c6, c7, c8 = st.columns(3)
    c6.metric("Total convênio", qtd_conv)
    c7.metric("Público", qtd_public)
    c8.metric("SUS", qtd_sus)

    st.caption(
        f"Destaques na tabela: vermelho = **EM ATRASO** · amarelo = **no prazo (em negociação)** com "
        f"**DATA PREVISTA** em ≤ {DAYS_THRESHOLD} dias"
    )

    # -------- Ordenação + Ícones + Formatação --------
    dff_sorted = (
        dff.sort_values(by=["REGIONAL", "OPERADORA"], kind="stable", na_position="last")
           .reset_index(drop=True)
    )

    # Recalcula máscaras com o DataFrame já ordenado (para alinhar com a tabela)
    s_sorted = dff_sorted["STATUS"].astype(str).str.strip().str.upper().fillna("")
    mask_no_prazo_labels_sorted = s_sorted.isin(no_prazo_labels)
    if int(mask_no_prazo_labels_sorted.sum()) == 0:
        mask_no_prazo_sorted = ~s_sorted.isin({"REALIZADO", "EM ATRASO", ""})
    else:
        mask_no_prazo_sorted = mask_no_prazo_labels_sorted

    dt_prev_sorted = pd.to_datetime(dff_sorted["DATA PREVISTA"], errors="coerce", dayfirst=True)
    delta_days_sorted = (dt_prev_sorted - today).dt.days
    mask_soon_sorted = dt_prev_sorted.notna() & (delta_days_sorted >= 0) & (delta_days_sorted <= DAYS_THRESHOLD)

    # Ícones leves (sem Styler)
    icons = np.where(s_sorted.eq("EM ATRASO"), "🔴",
             np.where(mask_no_prazo_sorted & mask_soon_sorted, "🟡", ""))

    dff_fmt = format_df_for_dashboard(dff_sorted)
    dff_fmt.insert(0, "⚠", icons)

    # Coluna 'id' no final apenas para exibição
    if "id" in dff_fmt.columns:
        col_order = [c for c in dff_fmt.columns if c != "id"] + ["id"]
        dff_fmt = dff_fmt[col_order]

    # --- Renderização rápida por padrão ---
    usar_destaque = st.toggle("Realçar linhas (pode ficar lento)", value=False)

    if usar_destaque:
        styled = (
            dff_fmt.style
                 .apply(row_highlight_style, axis=1)
                 .set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}])
        )
        st.dataframe(styled, use_container_width=True, height=520)
    else:
        st.dataframe(dff_fmt, use_container_width=True, height=520)
        # Cabeçalho em negrito mesmo sem Styler
        st.markdown(
            "<style>[data-testid='stDataFrame'] th {font-weight:700 !important;}</style>",
            unsafe_allow_html=True
        )

# REGIONAIS – EDIÇÃO
elif page == "Regionais (editar)":
    st.header("Edição por Regional (CRUD)")
    reg = st.selectbox("Escolha a regional", REGIONS, index=0)
    df = fetch_region(reg)

    st.caption("Edite os dados abaixo. Use o botão **Adicionar linha** para novos registros.")
    if st.button("➕ Adicionar linha"):
        blank = {c: None for c in ["id"] + UI_COLS}
        blank["REGIONAL"] = reg
        df = pd.concat([pd.DataFrame([blank]), df], ignore_index=True)

    # Coluna 'id' no final na grade de edição
    editor_cols = UI_COLS + ["id"]

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_order=editor_cols,
        disabled=["id", "REGIONAL"],
        column_config={
            "id": st.column_config.TextColumn(help="Identificador do registro (não editável)"),
            "% PREVISTO": st.column_config.NumberColumn(format="%.2f"),
            "% REALIZADO": st.column_config.NumberColumn(format="%.2f"),
            "TAXA": st.column_config.NumberColumn(format="%.2f"),
            "CONSULTAS": st.column_config.NumberColumn(format="%.2f"),
            "EXAMES": st.column_config.NumberColumn(format="%.2f"),
            "CIRURGIAS": st.column_config.NumberColumn(format="%.2f"),
            "HM": st.column_config.NumberColumn(format="%.2f"),
            "DATA DO CONTRATO": st.column_config.DateColumn(),
            "DATA ULTIMO REAJUSTE": st.column_config.DateColumn(),
            "DATA PREVISTA": st.column_config.DateColumn(),
            "DATA REALIZADA": st.column_config.DateColumn(),
            NEW_COL_CANON: st.column_config.DateColumn(),
        },
        hide_index=True,
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("💾 Salvar alterações na regional", type="primary"):
        try:
            sync_region(reg, edited)
            st.success(f"{reg}: alterações salvas.")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")

    if c2.button("↩️ Recarregar"):
        fetch_region.clear()
        st.rerun()

# IMPORTAR INICIAL – EXCEL → ARQUIVO
else:
    st.header("Importar inicial a partir do Excel (uma vez)")
    st.caption("Carregue o Excel atual com as abas AL, BA, PA, PE, RJ, DF, SC, SP. Após importar, a edição será **somente pelo app**.")

    def read_excel_file(file) -> pd.DataFrame:
        xls = pd.ExcelFile(file)
        dfs = []
        for reg in REGIONS:
            if reg not in xls.sheet_names:
                st.warning(f"Aba {reg} não encontrada; será ignorada.")
                continue
            dfr = pd.read_excel(xls, sheet_name=reg, dtype=object)

            # Sinônimos -> nome canônico
            rename_map = {src: NEW_COL_CANON for src in dfr.columns if src in IMPORT_SYNONYMS}
            if rename_map:
                dfr = dfr.rename(columns=rename_map)

            # Normaliza colunas / ordem
            for c in UI_COLS:
                if c not in dfr.columns:
                    dfr[c] = None
            dfr = dfr[UI_COLS].copy()

            # Datas
            for c in DATE_COLS:
                dfr[c] = to_date_series(dfr[c])

            # Percentuais
            for c in PCT_COLS:
                dfr[c] = dfr[c].apply(to_percent_0_100)

            # Decimais
            for c in DEC_COLS:
                dfr[c] = dfr[c].apply(to_decimal)

            # Texto
            for c in TEXT_COLS:
                dfr[c] = dfr[c].astype(str).str.strip().replace({"": None, "nan": None, "None": None})

            # Regional: força exatamente o nome da aba
            dfr["REGIONAL"] = str(reg).strip().upper()

            # Remove linhas 100% vazias (exceto OBS)
            mask_any = dfr.drop(columns=["OBS"]).notna().any(axis=1)
            dfr = dfr[mask_any]

            if not dfr.empty:
                dfs.append(dfr)

        dfs_nonempty = [x for x in dfs if not x.empty]
        return pd.concat(dfs_nonempty, ignore_index=True) if dfs_nonempty else pd.DataFrame(columns=UI_COLS)

    up = st.file_uploader("Envie o arquivo .xlsx", type=["xlsx"])
    if up is not None:
        with st.spinner("Lendo arquivo..."):
            df_import = read_excel_file(up)

        st.subheader("Prévia dos dados a importar")
        st.dataframe(df_import.head(100), use_container_width=True, height=400)
        st.write(f"Total a importar: **{len(df_import)}** linhas")

        if len(df_import):
            st.write("Pré-contagem por regional (import):")
            st.write(df_import["REGIONAL"].astype(str).str.strip().str.upper().value_counts())

        mode = st.radio("Modo de importação", ["Anexar (acrescenta)", "Substituir (apaga e reimporta)"])
        confirm = st.checkbox("Confirmo que esta importação é intencional")

        if st.button("🚀 Executar importação"):
            if not confirm:
                st.warning("Marque a caixa de confirmação para continuar.")
                st.stop()

            try:
                cur = csv_load()
                if mode.startswith("Substituir"):
                    base = df_import.copy()
                else:
                    base = pd.concat([cur, df_import], ignore_index=True)

                if "id" not in base.columns:
                    base.insert(0, "id", None)
                base["id"] = base["id"].apply(lambda x: x if (isinstance(x, str) and len(x) > 5) else str(uuid.uuid4()))

                csv_save(base)

                fetch_all.clear(); fetch_region.clear()

                st.success(f"Importação concluída. Total atual no arquivo: {len(csv_load())}")
                with st.expander("Prévia do arquivo após importação", expanded=True):
                    st.write(csv_load().groupby("REGIONAL").size().rename("qtd").reset_index())

                st.toast("Pronto! Abra a aba Dashboard para visualizar.", icon="✅")

            except Exception as e:
                st.error(f"Falha na importação: {e}")
