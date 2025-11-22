import streamlit as st
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt
from typing import List, Dict, Optional


# ---------- Daten laden ----------

def load_companies(csv_path: str = "dax40_companies.csv") -> pd.DataFrame:
    """
    Lädt die DAX40-Unternehmen aus der CSV.
    Erkennt automatisch Komma/Semikolon als Trennzeichen
    und zeigt zur Kontrolle die Spaltennamen in der Sidebar an.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    st.sidebar.write("CSV-Spalten:", list(df.columns))

    if "sector" not in df.columns:
        st.error(
            "In der CSV wurde keine Spalte namens **'sector'** gefunden.\n\n"
            f"Aktuelle Spalten sind: {list(df.columns)}"
        )
        st.stop()

    return df


# ---------- Dividenden-Helfer ----------

def infer_dividend_frequency(div_series: pd.Series) -> Optional[str]:
    """
    Heuristik zur Dividendenfrequenz:
    schaut sich die Anzahl Ausschüttungen im letzten Jahr an.
    """
    if div_series is None or div_series.empty:
        return None

    last_date = div_series.index.max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    recent = div_series[div_series.index >= one_year_ago]

    n = len(recent)
    if n == 0:
        return None
    if n >= 4:
        return "quarterly"
    if 2 <= n <= 3:
        return "semiannual"
    if n == 1:
        return "annual"
    return "irregular"


def calc_dividend_growth_5y(div_series: pd.Series) -> Optional[float]:
    """
    Berechnet das Dividendenwachstum (CAGR) über bis zu 5 Jahre.
    Nutzt Kalenderjahre (Summe der Dividenden pro Jahr).
    """
    if div_series is None or div_series.empty:
        return None

    yearly = div_series.groupby(div_series.index.year).sum().sort_index()
    if len(yearly) < 2:
        return None

    # max. 5 Jahre Fenster
    first_year = yearly.index[0]
    last_year = yearly.index[-1]
    if last_year - first_year > 5:
        first_year = last_year - 5
        yearly = yearly[yearly.index >= first_year]
        if len(yearly) < 2:
            return None

    first_val = float(yearly.iloc[0])
    last_val = float(yearly.iloc[-1])
    if first_val <= 0 or last_val <= 0:
        return None

    n_years = yearly.index[-1] - yearly.index[0]
    if n_years <= 0:
        return None

    cagr = (last_val / first_val) ** (1 / n_years) - 1
    return cagr * 100.0


# ---------- Kennzahlen & Kursdaten laden ----------

def fetch_metrics(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Holt Kursdaten (bis zu 5y Historie) + Fundamentaldaten für jeden Ticker.
    Berechnet u. a.:

    - last_price
    - change_1d, change_5d, change_1y
    - vol_30d, vol_1y
    - market_cap, pe, forward_pe, pb, ps, ev_ebitda
    - net_margin, operating_margin, roe, roa
    - dividend_yield, dividend_per_share, payout_ratio
    - dividend_growth_5y, dividend_frequency
    - debt_to_equity, net_debt_ebitda
    - history (Close-Serie, für Charts & SMAs)
    """
    result: Dict[str, Dict[str, Optional[float]]] = {}
    errors: List[str] = []

    for ticker in tickers:
        metrics: Dict[str, Optional[float]] = {
            "last_price": None,
            "change_1d": None,
            "change_5d": None,
            "change_1y": None,
            "vol_30d": None,
            "vol_1y": None,
            "market_cap": None,
            "pe_ratio": None,
            "forward_pe": None,
            "pb_ratio": None,
            "ps_ratio": None,
            "ev_ebitda": None,
            "net_margin": None,
            "operating_margin": None,
            "roe": None,
            "roa": None,
            "dividend_yield": None,
            "dividend_per_share": None,
            "payout_ratio": None,
            "dividend_growth_5y": None,
            "dividend_frequency": None,
            "debt_to_equity": None,
            "net_debt_ebitda": None,
            "history": None,
        }

        try:
            t = yf.Ticker(ticker)

            # --- Kurs-Historie: bis zu 5 Jahre ---
            hist = t.history(period="5y")
            if hist.empty:
                errors.append(f"{ticker}: keine Kursdaten erhalten.")
            else:
                close = hist["Close"]
                metrics["history"] = close

                last_price = float(close.iloc[-1])
                metrics["last_price"] = last_price

                rets = close.pct_change().dropna()

                # % Veränderungen
                def pct_change_from_days(days: int) -> Optional[float]:
                    if len(close) <= days:
                        # wenn zu wenige Daten: gegen ersten Wert vergleichen
                        base = float(close.iloc[0])
                    else:
                        base = float(close.iloc[-1 - days])
                    if base == 0:
                        return None
                    return (last_price - base) / base * 100.0

                metrics["change_1d"] = pct_change_from_days(1)
                metrics["change_5d"] = pct_change_from_days(5)
                # ~252 Handelstage ~ 1 Jahr
                metrics["change_1y"] = pct_change_from_days(252)

                # Volas
                if len(rets) >= 30:
                    metrics["vol_30d"] = float(rets.tail(30).std() * sqrt(252))
                if len(rets) >= 252:
                    metrics["vol_1y"] = float(rets.tail(252).std() * sqrt(252))
                elif len(rets) > 0:
                    metrics["vol_1y"] = float(rets.std() * sqrt(252))

            # --- Fundamentaldaten ---
            try:
                info = t.info

                metrics["market_cap"] = info.get("marketCap")
                metrics["pe_ratio"] = info.get("trailingPE")
                metrics["forward_pe"] = info.get("forwardPE")
                metrics["pb_ratio"] = info.get("priceToBook")
                metrics["ps_ratio"] = info.get("priceToSalesTrailing12Months")
                metrics["ev_ebitda"] = info.get("enterpriseToEbitda")

                metrics["net_margin"] = info.get("profitMargins")
                metrics["operating_margin"] = info.get("operatingMargins")
                metrics["roe"] = info.get("returnOnEquity")
                metrics["roa"] = info.get("returnOnAssets")

                metrics["dividend_yield"] = info.get("dividendYield")
                metrics["dividend_per_share"] = info.get("dividendRate")
                metrics["payout_ratio"] = info.get("payoutRatio")

                metrics["debt_to_equity"] = info.get("debtToEquity")
                total_debt = info.get("totalDebt")
                total_cash = info.get("totalCash")
                ebitda = info.get("ebitda")

                if (
                    total_debt is not None
                    and total_cash is not None
                    and ebitda not in (None, 0)
                ):
                    net_debt = float(total_debt) - float(total_cash)
                    metrics["net_debt_ebitda"] = float(net_debt) / float(ebitda)

            except Exception as e_info:
                errors.append(f"{ticker}: Fehler beim Laden von info – {e_info}")

            # --- Dividendenhistorie für Wachstum & Frequenz ---
            try:
                divs = t.dividends
                if divs is not None and not divs.empty:
                    metrics["dividend_frequency"] = infer_dividend_frequency(divs)
                    dg = calc_dividend_growth_5y(divs)
                    metrics["dividend_growth_5y"] = dg
            except Exception as e_div:
                errors.append(f"{ticker}: Fehler bei Dividendenhistorie – {e_div}")

        except Exception as e:
            errors.append(f"{ticker}: Allgemeiner Fehler – {e}")

        result[ticker] = metrics

    if errors:
        st.sidebar.error("Probleme beim Laden der Daten:")
        for msg in errors:
            st.sidebar.write("- ", msg)

    return result


# ---------- Graph aufbauen & zeichnen ----------

def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in df.iterrows():
        ticker = row["ticker_yahoo"]
        sector = row["sector"]
        price = row.get("last_price", None)

        if pd.isna(price):
            label_price = "n/a"
        else:
            label_price = f"{price:.2f} €"

        label = f"{ticker}\n{label_price}"

        G.add_node(
            ticker,
            label=label,
            name=row["name"],
            sector=sector,
        )

    for sector in df["sector"].dropna().unique():
        same_sector = df[df["sector"] == sector]["ticker_yahoo"].tolist()
        for i in range(len(same_sector)):
            for j in range(i + 1, len(same_sector)):
                G.add_edge(same_sector[i], same_sector[j], relation="same_sector")

    return G


def draw_graph(G: nx.Graph):
    if len(G.nodes) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)

    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------- Style-Helfer ----------

def colorize_change(val):
    if pd.isna(val):
        return ""
    color = "green" if val > 0 else "red" if val < 0 else "black"
    return f"color: {color}"


# ---------- Streamlit App ----------

def main():
    st.title("DAX40 Explorer – Mini-Tool mit Fundamentaldaten")

    df = load_companies()

    # --- Filter ---
    st.sidebar.header("Filter")
    sector_options = ["Alle"] + sorted(df["sector"].dropna().unique().tolist())
    selected_sector = st.sidebar.selectbox("Sektor", sector_options)
    search_text = st.sidebar.text_input(
        "Suche nach Firmenname oder Ticker (optional)"
    ).strip()

    # --- Daten laden ---
    st.subheader("Schritt 1: Kurse, Performance & Fundamentaldaten laden")
    if st.button("Daten laden / aktualisieren"):
        tickers = df["ticker_yahoo"].tolist()
        metrics = fetch_metrics(tickers)

        def map_metric(col_name: str):
            return df["ticker_yahoo"].map(
                lambda t: metrics.get(t, {}).get(col_name)
            )

        # Kurs & Performance
        df["last_price"] = map_metric("last_price")
        df["change_1d"] = map_metric("change_1d")
        df["change_5d"] = map_metric("change_5d")
        df["change_1y"] = map_metric("change_1y")
        df["vol_30d"] = map_metric("vol_30d")
        df["vol_1y"] = map_metric("vol_1y")

        # Bewertung & Profitabilität
        df["market_cap"] = map_metric("market_cap")
        df["pe_ratio"] = map_metric("pe_ratio")
        df["forward_pe"] = map_metric("forward_pe")
        df["pb_ratio"] = map_metric("pb_ratio")
        df["ps_ratio"] = map_metric("ps_ratio")
        df["ev_ebitda"] = map_metric("ev_ebitda")
        df["net_margin"] = map_metric("net_margin")
        df["operating_margin"] = map_metric("operating_margin")
        df["roe"] = map_metric("roe")
        df["roa"] = map_metric("roa")

        # Dividende & Verschuldung
        df["dividend_yield"] = map_metric("dividend_yield")
        df["dividend_per_share"] = map_metric("dividend_per_share")
        df["payout_ratio"] = map_metric("payout_ratio")
        df["dividend_growth_5y"] = map_metric("dividend_growth_5y")
        df["dividend_frequency"] = map_metric("dividend_frequency")
        df["debt_to_equity"] = map_metric("debt_to_equity")
        df["net_debt_ebitda"] = map_metric("net_debt_ebitda")

        # für spätere Nutzung
        st.session_state["metrics"] = metrics
        st.session_state["df_with_metrics"] = df.copy()

        st.success("Daten aktualisiert!")
    else:
        if "df_with_metrics" in st.session_state:
            df = st.session_state["df_with_metrics"]
        else:
            # leere Spalten initialisieren
            for col in [
                "last_price", "change_1d", "change_5d", "change_1y",
                "vol_30d", "vol_1y", "market_cap", "pe_ratio", "forward_pe",
                "pb_ratio", "ps_ratio", "ev_ebitda", "net_margin",
                "operating_margin", "roe", "roa", "dividend_yield",
                "dividend_per_share", "payout_ratio", "dividend_growth_5y",
                "dividend_frequency", "debt_to_equity", "net_debt_ebitda",
            ]:
                if col not in df.columns:
                    df[col] = None

    # --- Filter anwenden ---
    df_view = df.copy()
    if selected_sector != "Alle":
        df_view = df_view[df_view["sector"] == selected_sector]

    if search_text:
        mask = df_view["name"].str.contains(search_text, case=False, na=False) | \
               df_view["ticker_yahoo"].str.contains(search_text, case=False, na=False)
        df_view = df_view[mask]

    # --- Tabs: Overview, Fundamentals ---
    tab_overview, tab_funda = st.tabs(["Überblick", "Fundamentaldaten"])

    with tab_overview:
        st.subheader("Überblick: Kurs & Performance")

        overview_cols = [
            "name",
            "ticker_yahoo",
            "sector",
            "last_price",
            "change_1d",
            "change_5d",
            "change_1y",
            "vol_30d",
            "vol_1y",
        ]

        styled_overview = (
            df_view[overview_cols]
            .style.format(
                {
                    "last_price": "{:.2f}",
                    "change_1d": "{:+.2f}%",
                    "change_5d": "{:+.2f}%",
                    "change_1y": "{:+.2f}%",
                    "vol_30d": "{:.2f}",
                    "vol_1y": "{:.2f}",
                },
                na_rep="–",
            )
            .applymap(colorize_change, subset=["change_1d", "change_5d", "change_1y"])
        )

        st.dataframe(styled_overview, use_container_width=True)

    with tab_funda:
        st.subheader("Fundamentaldaten")

        funda_cols = [
            "name",
            "ticker_yahoo",
            "market_cap",
            "pe_ratio",
            "forward_pe",
            "pb_ratio",
            "ps_ratio",
            "ev_ebitda",
            "net_margin",
            "operating_margin",
            "roe",
            "roa",
            "dividend_yield",
            "dividend_per_share",
            "payout_ratio",
            "dividend_growth_5y",
            "dividend_frequency",
            "debt_to_equity",
            "net_debt_ebitda",
        ]

        styled_funda = (
            df_view[funda_cols]
            .style.format(
                {
                    "market_cap": "{:,.0f}",
                    "pe_ratio": "{:.2f}",
                    "forward_pe": "{:.2f}",
                    "pb_ratio": "{:.2f}",
                    "ps_ratio": "{:.2f}",
                    "ev_ebitda": "{:.2f}",
                    "net_margin": "{:.2f}%",
                    "operating_margin": "{:.2f}%",
                    "roe": "{:.2f}%",
                    "roa": "{:.2f}%",
                    "dividend_yield": "{:.2f}%",
                    "dividend_per_share": "{:.2f}",
                    "payout_ratio": "{:.2f}%",
                    "dividend_growth_5y": "{:.2f}%",
                    "debt_to_equity": "{:.2f}",
                    "net_debt_ebitda": "{:.2f}",
                },
                na_rep="–",
            )
        )

        styled_funda = styled_funda.applymap(
            colorize_change,
            subset=[
                "net_margin",
                "operating_margin",
                "roe",
                "roa",
                "dividend_yield",
                "dividend_growth_5y",
            ],
        )

        st.dataframe(styled_funda, use_container_width=True)

    # --- Mini-Chart / Sparkline + SMA ---
    st.subheader("Schritt 3: Mini-Kurscharts (Sparkline + SMAs)")

    if "metrics" in st.session_state and st.session_state["metrics"]:
        available_tickers = df_view["ticker_yahoo"].tolist()
        if available_tickers:
            selected_ticker = st.selectbox(
                "Wähle einen Ticker für den Mini-Chart:",
                options=available_tickers,
            )

            period_choice = st.selectbox(
                "Zeitraum",
                options=["2 Monate", "6 Monate", "1 Jahr", "5 Jahre"],
                index=0,
            )

            show_smas = st.checkbox("SMA 20/50/200 anzeigen", value=True)

            metrics_store = st.session_state["metrics"]
            series = metrics_store.get(selected_ticker, {}).get("history", None)

            if series is not None and not series.empty:
                # Fenstergröße abhängig vom Zeitraum
                if period_choice == "2 Monate":
                    window = 60
                elif period_choice == "6 Monate":
                    window = 130
                elif period_choice == "1 Jahr":
                    window = 252
                else:  # "5 Jahre"
                    window = len(series)

                window = min(window, len(series))

                spark = series.tail(window).to_frame(name="Kurs")

                if show_smas:
                    spark["SMA20"] = spark["Kurs"].rolling(window=20).mean()
                    spark["SMA50"] = spark["Kurs"].rolling(window=50).mean()
                    spark["SMA200"] = spark["Kurs"].rolling(window=200).mean()

                spark.index.name = "Datum"
                st.line_chart(spark)
            else:
                st.info("Für diesen Ticker ist keine Kurs-Historie verfügbar.")
        else:
            st.info("Keine Firmen für diese Filtereinstellung.")
    else:
        st.info("Bitte zuerst oben auf **'Daten laden / aktualisieren'** klicken.")

    # --- Graph-Ansicht ---
    st.subheader("Schritt 4: Graph-Ansicht (Mindmap-light)")
    st.write(
        "Hier siehst du die ausgewählten Firmen als Knoten. "
        "Kanten verbinden Firmen im gleichen Sektor."
    )

    if df_view.empty:
        st.info("Keine Firmen für diese Filtereinstellung gefunden.")
        return

    G = build_graph(df_view)
    fig = draw_graph(G)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("Graph konnte nicht gezeichnet werden (keine Knoten).")


if __name__ == "__main__":
    main()
