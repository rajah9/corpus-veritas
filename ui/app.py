"""
ui/app.py
Milestone 8: Streamlit prototype for corpus-veritas.

Two interaction modes:

  Chat        Plain-English exploration. Every response includes
              confidence tier, provenance, and audit entry ID inline.
              Sources cited with document UUIDs.

  Structured  Four tabs:
              - Timeline: entity + date-range query, chronological results
              - Relationship: entity pair graph traversal
              - Deletion Report: latest gap report viewer
              - Entity Lookup: search the entity registry

Run locally:
    streamlit run ui/app.py

Environment variables (same as api/handler.py):
    API_ENDPOINT    corpus-veritas API Gateway URL.
                    If not set, app calls the pipeline directly
                    (development mode -- requires all AWS credentials).

See infrastructure/DEPLOYMENT.md for API deployment instructions.
See CONSTITUTION.md for ethical constraints.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_ENDPOINT: str = os.environ.get("API_ENDPOINT", "")
APP_TITLE = "corpus-veritas"
APP_SUBTITLE = "Epstein Document Analysis — graded by evidence, governed by ethics"

# Confidence tier colour coding for display
_TIER_COLOURS: dict[str, str] = {
    "CONFIRMED":     "#2d7d46",  # green
    "CORROBORATED":  "#1a6891",  # blue
    "INFERRED":      "#7d6b2d",  # amber
    "SINGLE_SOURCE": "#7d4a2d",  # orange
    "SPECULATIVE":   "#7d2d2d",  # red
}

_TIER_LABELS: dict[str, str] = {
    "CONFIRMED":     "✓ Confirmed",
    "CORROBORATED":  "◎ Corroborated",
    "INFERRED":      "~ Inferred",
    "SINGLE_SOURCE": "! Single source",
    "SPECULATIVE":   "? Speculative",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post_query(payload: dict) -> dict | None:
    """POST /query to the API and return the parsed response body."""
    if not API_ENDPOINT:
        st.error(
            "API_ENDPOINT is not set. Set the environment variable to the "
            "corpus-veritas API Gateway URL and restart the app."
        )
        return None
    try:
        resp = requests.post(
            f"{API_ENDPOINT}/query",
            json=payload,
            timeout=30,
        )
        return resp.json()
    except requests.exceptions.Timeout:
        st.error("The query timed out. Try a more specific query or reduce top_k.")
        return None
    except Exception as exc:
        st.error(f"API request failed: {exc}")
        return None


def _get_gap_report(version: str = "latest", public: bool = True) -> dict | None:
    """GET /gap-report from the API."""
    if not API_ENDPOINT:
        st.error("API_ENDPOINT is not set.")
        return None
    try:
        resp = requests.get(
            f"{API_ENDPOINT}/gap-report",
            params={"version": version, "public": str(public).lower()},
            timeout=15,
        )
        return resp.json()
    except Exception as exc:
        st.error(f"Failed to retrieve gap report: {exc}")
        return None


def _get_entity(canonical_name: str, entity_type: str = "PERSON") -> dict | None:
    """GET /entity/{canonical_name} from the API."""
    if not API_ENDPOINT:
        st.error("API_ENDPOINT is not set.")
        return None
    try:
        import urllib.parse
        encoded = urllib.parse.quote(canonical_name, safe="")
        resp = requests.get(
            f"{API_ENDPOINT}/entity/{encoded}",
            params={"entity_type": entity_type},
            timeout=10,
        )
        return resp.json()
    except Exception as exc:
        st.error(f"Entity lookup failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _render_answer(result: dict) -> None:
    """Render a query result with confidence tier, provenance, and audit ID."""
    tier = result.get("lowest_tier", "")
    colour = _TIER_COLOURS.get(tier, "#666666")
    tier_label = _TIER_LABELS.get(tier, tier)

    st.markdown(
        f'<span style="color:{colour};font-weight:bold">{tier_label}</span>',
        unsafe_allow_html=True,
    )

    answer = result.get("answer", "")
    if not answer:
        st.warning("No answer was returned for this query.")
        return

    st.markdown(answer)

    # Guardrail flags
    flags = []
    if result.get("victim_scan_triggered"):
        flags.append("🛡 Victim identity protection applied")
    if result.get("inference_downgraded"):
        flags.append("⚠ Inference suppressed — convergence threshold not met")
    if result.get("confidence_violation"):
        flags.append("📊 Confidence language corrected")
    if result.get("creative_content_suppressed"):
        flags.append("🚫 Speculative content suppressed")
    if flags:
        for flag in flags:
            st.caption(flag)

    # Provenance footer
    audit_id = result.get("audit_entry_id", "")
    chunk_count = result.get("chunk_count", 0)
    retrieved_at = result.get("retrieved_at", "")
    st.caption(
        f"📋 {chunk_count} source chunk(s) retrieved · "
        f"Audit ID: `{audit_id}` · {retrieved_at[:19] if retrieved_at else ''}"
    )


def _render_error(result: dict) -> None:
    """Render an API error response."""
    status = result.get("statusCode", 500)
    error = result.get("error", "error")
    message = result.get("message", str(result))

    if status == 503:
        st.error(f"⛔ Service unavailable: {message}")
    elif status == 400:
        st.warning(f"Invalid request: {message}")
    elif status == 403:
        st.warning(f"🛡 {message}")
    else:
        st.error(f"Error ({status}): {message}")


# ---------------------------------------------------------------------------
# Mode: Chat
# ---------------------------------------------------------------------------

def render_chat_mode() -> None:
    st.markdown("### Ask a question about the Epstein documents")
    st.caption(
        "Every response is graded by confidence tier and provenance. "
        "Victim identities are suppressed. All queries are audited."
    )

    query_text = st.text_area(
        "Your question",
        placeholder="e.g. What do the documents say about the relationship between Epstein and Maxwell?",
        height=100,
        key="chat_query",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query_type = st.selectbox(
            "Query type",
            ["PROVENANCE", "TIMELINE", "INFERENCE", "RELATIONSHIP"],
            key="chat_query_type",
        )
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=10, key="chat_top_k")

    if st.button("Search", key="chat_submit", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving documents and synthesising answer…"):
            result = _post_query({
                "query_text": query_text,
                "query_type": query_type,
                "top_k": top_k,
            })

        if result is None:
            return

        if "statusCode" in result and result["statusCode"] != 200:
            _render_error(result)
        else:
            _render_answer(result)

    st.divider()
    st.caption(
        "⚠️ This system analyses publicly released documents. It does not make "
        "legal determinations. All inferences require multi-source corroboration. "
        "See the [Constitution](https://github.com/corpus-veritas/CONSTITUTION.md) "
        "for ethical constraints."
    )


# ---------------------------------------------------------------------------
# Mode: Structured — Timeline tab
# ---------------------------------------------------------------------------

def render_timeline_tab() -> None:
    st.markdown("#### Timeline Query")
    st.caption("Find documented events involving named individuals within a date range.")

    entity_input = st.text_input(
        "Entity name(s) (comma-separated)",
        placeholder="e.g. jeffrey epstein, ghislaine maxwell",
        key="timeline_entities",
    )
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From date", key="timeline_from", value=None)
    with col2:
        date_to = st.date_input("To date", key="timeline_to", value=None)

    query_text = st.text_input(
        "Optional: refine with a question",
        placeholder="e.g. What meetings are documented?",
        key="timeline_query",
    )

    if st.button("Search Timeline", key="timeline_submit"):
        if not query_text.strip():
            query_text = f"Timeline of events involving {entity_input}"

        entity_names = [e.strip() for e in entity_input.split(",") if e.strip()]
        payload = {
            "query_text": query_text,
            "query_type": "TIMELINE",
            "entity_names": entity_names or None,
            "date_from": str(date_from) if date_from else None,
            "date_to":   str(date_to)   if date_to   else None,
        }

        with st.spinner("Searching timeline…"):
            result = _post_query(payload)

        if result and result.get("statusCode", 200) == 200:
            _render_answer(result)
        elif result:
            _render_error(result)


# ---------------------------------------------------------------------------
# Mode: Structured — Relationship tab
# ---------------------------------------------------------------------------

def render_relationship_tab() -> None:
    st.markdown("#### Relationship Query")
    st.caption(
        "Find documented connections between two named individuals. "
        "Graph traversal is used when available."
    )

    col1, col2 = st.columns(2)
    with col1:
        entity_a = st.text_input("Person A", placeholder="e.g. jeffrey epstein", key="rel_a")
    with col2:
        entity_b = st.text_input("Person B", placeholder="e.g. prince andrew", key="rel_b")

    if st.button("Find Connections", key="rel_submit"):
        if not entity_a.strip() or not entity_b.strip():
            st.warning("Please enter both entity names.")
            return

        payload = {
            "query_text": f"What documented connections exist between {entity_a} and {entity_b}?",
            "query_type": "RELATIONSHIP",
            "entity_names": [entity_a.strip(), entity_b.strip()],
        }

        with st.spinner("Traversing relationship graph…"):
            result = _post_query(payload)

        if result and result.get("statusCode", 200) == 200:
            _render_answer(result)
        elif result:
            _render_error(result)


# ---------------------------------------------------------------------------
# Mode: Structured — Deletion Report tab
# ---------------------------------------------------------------------------

def render_deletion_tab() -> None:
    st.markdown("#### Deletion Detection Report")
    st.caption(
        "Documented gaps between the DOJ index and the public release. "
        "Methodology: NPR February 2026 / WSJ March 2026."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        version = st.text_input("Release version", value="latest", key="del_version")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load Report", key="del_submit"):
            with st.spinner("Loading gap report…"):
                result = _get_gap_report(version=version, public=True)

            if result is None:
                return
            if result.get("statusCode", 200) != 200:
                _render_error(result)
                return

            report_md = result.get("report", "")
            if report_md:
                st.markdown(report_md)
                st.caption(
                    f"Version: `{result.get('version', version)}` · "
                    f"Public report (victim identities suppressed)"
                )
            else:
                st.info("No report content returned.")


# ---------------------------------------------------------------------------
# Mode: Structured — Entity Lookup tab
# ---------------------------------------------------------------------------

def render_entity_tab() -> None:
    st.markdown("#### Entity Registry Lookup")
    st.caption("Search for named individuals and organisations in the entity registry.")

    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input(
            "Canonical name",
            placeholder="e.g. jeffrey epstein",
            key="entity_name",
        )
    with col2:
        entity_type = st.selectbox(
            "Type",
            ["PERSON", "ORGANIZATION", "LOCATION"],
            key="entity_type",
        )

    if st.button("Lookup", key="entity_submit"):
        if not name.strip():
            st.warning("Please enter a name.")
            return

        with st.spinner("Looking up entity…"):
            result = _get_entity(name.strip().lower(), entity_type)

        if result is None:
            return
        if result.get("statusCode", 200) != 200:
            _render_error(result)
            return

        st.markdown(f"**{result.get('canonical_name', name)}** ({entity_type})")

        surface_forms = result.get("surface_forms", [])
        if surface_forms:
            st.caption(f"Surface forms: {', '.join(surface_forms)}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
        with col2:
            doc_count = len(result.get("document_uuids", []))
            st.metric("Documents", doc_count)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.title(APP_TITLE)
        st.caption(APP_SUBTITLE)
        st.divider()

        mode = st.radio(
            "Mode",
            ["Chat", "Structured View"],
            key="mode",
        )

        st.divider()
        st.caption("**Confidence tiers**")
        for tier, label in _TIER_LABELS.items():
            colour = _TIER_COLOURS[tier]
            st.markdown(
                f'<span style="color:{colour}">{label}</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        if API_ENDPOINT:
            st.success(f"API connected: `{API_ENDPOINT[:40]}…`")
        else:
            st.warning("API_ENDPOINT not set")

        st.caption(
            "⚖ corpus-veritas · Apache 2.0 · "
            "[Constitution](https://github.com/corpus-veritas/CONSTITUTION.md)"
        )

    # Main content
    if mode == "Chat":
        render_chat_mode()
    else:
        tab_timeline, tab_rel, tab_del, tab_entity = st.tabs([
            "📅 Timeline",
            "🔗 Relationship",
            "🕳 Deletion Report",
            "👤 Entity Lookup",
        ])
        with tab_timeline:  render_timeline_tab()
        with tab_rel:       render_relationship_tab()
        with tab_del:       render_deletion_tab()
        with tab_entity:    render_entity_tab()


if __name__ == "__main__":
    main()
