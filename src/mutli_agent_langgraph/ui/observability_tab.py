# Compare outputs for the same prompt_hash across models (tab version)
import os, tempfile, pandas as pd, streamlit as st
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

# Defaults (will be overridden by cfg if present)
DEFAULT_TRACKING_URI = "./mlruns"
DEFAULT_EXPERIMENT   = "QEA_Cognitive"
ANSWER_ARTIFACT      = "outputs/ui_streamed_answer.txt"
ALT_ANSWER_ARTIFACT  = "outputs/node_last_answer.txt"
PROMPT_ARTIFACT_TPL  = "prompts/{hash}.txt"
ALT_PROMPT_PREVIEW   = "prompts/ui_prompt_preview.txt"

def _client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)

def _exp_id(client: MlflowClient, name: str) -> Optional[str]:
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else None
def _search_by_hash(client, exp_id, prompt_hash, max_results=50):
    # 1) Search by param
    runs = client.search_runs(
        [exp_id],
        filter_string=f"params.prompt_hash = '{prompt_hash}'",
        order_by=["attributes.start_time DESC"],
        max_results=max_results
    )
    if runs: 
        return runs
    # 2) Fallback: search by tag
    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.prompt_hash = '{prompt_hash}'",
        order_by=["attributes.start_time DESC"],
        max_results=max_results
    )
    return runs


def _download_text_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> Optional[str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, artifact_path, tmp)
            if os.path.isdir(local):
                return None
            with open(local, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return None

def render_observe_tab(cfg=None):
    # Pull defaults from your Config if provided
    tracking_uri = DEFAULT_TRACKING_URI
    experiment   = DEFAULT_EXPERIMENT
    if cfg:
        try:
            tracking_uri = cfg.tracking("mlflow_tracking_uri", fallback=tracking_uri)
            experiment   = cfg.tracking("experiment_name", fallback=experiment)
        except Exception:
            pass

    st.subheader("🔎 Observability — Compare Models by Prompt Hash")
    with st.sidebar:
        st.markdown("### Observability Settings")
        tracking_uri = st.text_input("MLflow Tracking URI", value=tracking_uri)
        experiment   = st.text_input("Experiment Name", value=experiment)
        max_results  = st.number_input("Max runs to fetch", min_value=5, max_value=200, value=50, step=5)

    prompt_hash = st.text_input("Enter `prompt_hash`", placeholder="e.g., e8d1f23a4c5b").strip()
    if not prompt_hash:
        st.info("Enter a `prompt_hash` to load matching runs. (Run the same query on 2+ models first.)")
        return

    client = _client(tracking_uri)
    exp_id = _exp_id(client, experiment)
    if not exp_id:
        st.error(f"Experiment '{experiment}' not found at {tracking_uri}.")
        return

    runs = _search_by_hash(client, exp_id, prompt_hash, max_results)
    if not runs:
        st.warning(f"No runs found with prompt_hash = {prompt_hash}")
        return

    rows = []
    for r in runs:
        p, m = r.data.params, r.data.metrics
        rows.append({
            "run_id": r.info.run_id,
            "status": r.info.status,
            "llm_model": p.get("llm_model", "unknown"),
            "temperature": p.get("llm_temperature", "n/a"),
            "plan_generation": p.get("plan_generation", ""),
            "ui_latency_sec": m.get("ui_stream_latency_sec"),
            "node_latency_sec": m.get("node_total_latency_sec"),
            "input_tokens": m.get("llm_input_tokens"),
            "output_tokens": m.get("llm_output_tokens"),
            "total_tokens": m.get("llm_total_tokens"),
            "tc_valid_ratio": m.get("tc_valid_ratio"),
            "tc_count": m.get("tc_count"),
            "start_time": r.info.start_time,
        })

    df = pd.DataFrame(rows).sort_values("start_time", ascending=False)

    st.markdown("#### Matching Runs")
    st.dataframe(
        df[["run_id","status","llm_model","temperature","plan_generation","ui_latency_sec","node_latency_sec","total_tokens","tc_valid_ratio","tc_count"]],
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("#### Side-by-side Answers")

    selected_models = st.multiselect(
        "Filter by model (optional)",
        sorted(df["llm_model"].dropna().unique().tolist()),
        default=sorted(df["llm_model"].dropna().unique().tolist())
    )

    for _, row in df.iterrows():
        if selected_models and row["llm_model"] not in selected_models:
            continue

        run_id = row["run_id"]
        header = f"{row['llm_model']}  • run={run_id[:8]}… • temp={row['temperature']} • latency={row['ui_latency_sec']}s • tokens={row['total_tokens']}"
        with st.expander(header, expanded=False):
            prompt_text = _download_text_artifact(client, run_id, PROMPT_ARTIFACT_TPL.format(hash=prompt_hash))
            if not prompt_text:
                prompt_text = _download_text_artifact(client, run_id, ALT_PROMPT_PREVIEW) or "(prompt artifact not found)"

            answer_text = _download_text_artifact(client, run_id, ANSWER_ARTIFACT)
            if not answer_text:
                answer_text = _download_text_artifact(client, run_id, ALT_ANSWER_ARTIFACT) or "(answer artifact not found)"

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Prompt")
                st.code(prompt_text, language="markdown")
            with c2:
                st.caption("Answer")
                st.code(answer_text, language="markdown")

            mcols = st.columns(5)
            mcols[0].metric("UI Latency (s)", value=row.get("ui_latency_sec"))
            mcols[1].metric("Node Latency (s)", value=row.get("node_latency_sec"))
            mcols[2].metric("Total Tokens", value=row.get("total_tokens"))
            mcols[3].metric("TC Valid Ratio", value=row.get("tc_valid_ratio"))
            mcols[4].metric("# Test Cases", value=row.get("tc_count"))
