"""Gradio UI for the AgentEvalOps service (mounted at /ui by code/service/app.py).

Tabs
  Ask          task input, bundled examples, optional fault injection or model swap
               (drift demo); shows answer, confidence, latency, trace id, version,
               and every step with its tool call, arguments and output
  Monitoring   live label-free signals vs their validation reference, detector
               state and drift alerts; a button replays bundled shifted tasks
  Validation   reported metrics next to a live run of the fixed gold subset
  About        git commit, registry version, configuration and model digest

Only system fonts are used, so the page makes no request to a font CDN and works
with the internet disconnected.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import gradio as gr

from code.evaluation.data import load_split

EXAMPLE_SPLITS = ("validation", "test")


def _examples() -> list[tuple[str, str]]:
    out = []
    for split in EXAMPLE_SPLITS:
        for item in sorted(load_split(split), key=lambda i: i.task_id):
            if item.item_type == "base":
                out.append((f"[{split}] {item.task_id}", item.task.prompt))
    return out


def ui_theme() -> gr.themes.Base:
    """System fonts only, so the page loads no font from a CDN (Gradio 6 takes the theme at mount time)."""
    return gr.themes.Default(font=["system-ui", "Segoe UI", "Arial", "sans-serif"],
                             font_mono=["Consolas", "Menlo", "monospace"])


def build_ui(service: Callable[[], Any]) -> gr.Blocks:
    from code.service.app import FAULT_PROFILES, PredictRequest

    examples = _examples()
    example_map = dict(examples)

    def ask(task: str, fault: str, swap: bool) -> tuple[Any, ...]:
        svc = service()
        if not task.strip():
            return ("Enter a task.", "", "", "", "", [], "")
        res = svc.predict(PredictRequest(task=task, fault=fault, model="qwen2.5:1.5b" if swap else None))
        steps = [[s["index"], s["action"], s["tool"] or "", json.dumps(s["arguments"]) if s["arguments"] else "",
                  json.dumps(s["output"], default=str)[:300] if s["output"] is not None else "", s["error"] or "",
                  round((s["latency_ms"] or 0) / 1000, 2)] for s in res["steps"]]
        grade = ""
        if res["graded"]:
            grade = f"{'correct' if res['correct'] else 'wrong'} (gold: {res['gold_answer'] or 'unanswerable'})"
        alerts = "; ".join(f"{a['signal']}/{a['detector']}" for a in res["alerts"]) or "none"
        conf = "n/a" if res["confidence"] is None else f"{res['confidence']:.2f}"
        return (res["answer"] or "(no answer)", conf, f"{res['latency_ms'] / 1000:.1f} s",
                f"{res['version']} / {res['model']} / commit {res['git_commit'][:10]}", res["trace_id"], steps,
                f"{grade}  |  drift alerts raised by this request: {alerts}")

    def monitoring_view() -> tuple[Any, Any]:
        snap = service().monitor.snapshot()
        table = [[s, snap["rolling"][s], snap["reference"][s], "yes" if s in snap["monitored_signals"] else ""]
                 for s in snap["rolling"]]
        summary = (f"Requests: {snap['requests']}  |  rolling success on graded tasks: {snap['rolling_success_graded']}"
                   f"  |  drift detected: {'YES' if snap['drift'] else 'no'}\n"
                   f"Detector statistics: {snap['detector_state']}\nRecent alerts: {snap['alerts']}")
        return table, summary

    def replay_shifted(n: int, fault_filter: str) -> tuple[Any, Any]:
        svc = service()
        items = [i for i in sorted(load_split("shifted_test"), key=lambda i: i.task_id)
                 if fault_filter == "any" or i.fault_label == fault_filter][: int(n)]
        from code.agent.loop import run_agent
        from code.augmentation.fault_injection import FaultInjector

        for item in items:
            with svc.lock:
                with FaultInjector(item.faults):
                    run = run_agent(item.task, svc.cfg, client=svc.client, tracer=svc.tracer)
            svc.monitor.update(run.summary(), graded=True)
        return monitoring_view()

    def validation_run() -> tuple[str, Any]:
        res = service().run_validation()
        rows = [[r["task_id"], r["answer"], r["correct"], r["reported_correct"]] for r in res["rows"]]
        text = (f"Live success on the fixed subset: {res['live_success']} (reported: {res['reported_success']}); "
                f"per-task match with reported: {res['matches_reported']}; {res['wall_s']} s")
        return text, rows

    def reported_text() -> str:
        rep = service().reported()
        if not rep:
            return "No reported values bundled."
        return json.dumps({k: v for k, v in rep.items() if k != "ui_validation_subset"}, indent=2)

    with gr.Blocks(title="AgentEvalOps", analytics_enabled=False) as demo:
        gr.Markdown("## AgentEvalOps: tool-using agent, local CPU inference")
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    pick = gr.Dropdown([e[0] for e in examples], label="Bundled examples", value=None)
                    task = gr.Textbox(label="Task", lines=4)
                    with gr.Row():
                        fault = gr.Dropdown(list(FAULT_PROFILES), value="none", label="Inject tool fault (drift demo)")
                        swap = gr.Checkbox(label="Swap model to qwen2.5:1.5b (drift demo)")
                    go = gr.Button("Run agent", variant="primary")
                with gr.Column(scale=2):
                    answer = gr.Textbox(label="Answer")
                    with gr.Row():
                        confidence = gr.Textbox(label="Confidence (P correct)")
                        latency = gr.Textbox(label="Latency")
                    version = gr.Textbox(label="Version / model / commit")
                    trace = gr.Textbox(label="MLflow trace id")
                    grading = gr.Textbox(label="Grading and monitoring")
            steps = gr.Dataframe(headers=["step", "action", "tool", "arguments", "output", "error", "seconds"],
                                 label="Steps and tool calls", wrap=True)
            pick.change(lambda k: example_map.get(k, ""), pick, task)
            go.click(ask, [task, fault, swap], [answer, confidence, latency, version, trace, steps, grading])
        with gr.Tab("Monitoring"):
            mon_table = gr.Dataframe(headers=["signal", "rolling mean", "validation reference", "monitored"],
                                     label="Label-free signals (last 25 requests)")
            mon_text = gr.Textbox(label="Status", lines=4)
            with gr.Row():
                refresh = gr.Button("Refresh")
                n = gr.Number(value=5, precision=0, label="Shifted test tasks to replay")
                ff = gr.Dropdown(["any", "transient_failure", "information_outage", "slow_response", "format_change"],
                                 value="information_outage", label="Fault type")
                replay = gr.Button("Replay shifted traffic")
                reset = gr.Button("Reset monitor")
            refresh.click(monitoring_view, None, [mon_table, mon_text])
            replay.click(replay_shifted, [n, ff], [mon_table, mon_text])
            reset.click(lambda: (service().monitor.reset(), *monitoring_view())[1:], None, [mon_table, mon_text])
        with gr.Tab("Validation"):
            rep = gr.Code(label="Reported values (from the report)", language="json")
            run_val = gr.Button("Run fixed gold subset now")
            val_text = gr.Textbox(label="Result")
            val_rows = gr.Dataframe(headers=["task", "answer", "correct (live)", "correct (reported)"])
            run_val.click(validation_run, None, [val_text, val_rows])
        with gr.Tab("About"):
            about = gr.Code(language="json", label="Version")
        demo.load(lambda: json.dumps(service().version_info(), indent=2), None, about)
        demo.load(reported_text, None, rep)
        demo.load(monitoring_view, None, [mon_table, mon_text])
    return demo
