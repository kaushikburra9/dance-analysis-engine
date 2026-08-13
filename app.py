"""
Dance Analysis — local web app (Gradio) for testing videos on both engines.

Run:   source venv/bin/activate && python app.py
Then open the printed http://127.0.0.1:7860 in your browser.

Tab 1  Choreography Compare : student vs reference dancer (compare_dance.py)
Tab 2  Multi-Dancer Analysis: tracking + formation consistency (analyze_run.py)
"""
import os, sys, time, subprocess, re
import gradio as gr

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_runs")
os.makedirs(RUNS, exist_ok=True)
PY = sys.executable


def _run_dir(tag):
    d = os.path.join(RUNS, f"{tag}_{int(time.time())}")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------
# Tab 1 — choreography comparison
# ----------------------------------------------------------------------------
def compare(student, reference, margin, freeze, progress=gr.Progress()):
    if not student or not reference:
        return None, None, None, "⚠️ Upload BOTH a student and a reference clip."
    import compare_dance as cd
    try:
        progress(0.1, desc="Extracting student pose…")
        S = cd.extract_pose(student)
        progress(0.35, desc="Extracting reference pose…")
        R = cd.extract_pose(reference)
        progress(0.55, desc="Aligning timing + flagging…")
        wp, events, lag = cd.align_and_flag(S, R, margin, 0.4)
        steps = cd.group_steps(events)
        state = ("in sync with" if abs(lag) < 0.05 else
                 "AHEAD of" if lag > 0 else "BEHIND")
        d = _run_dir("compare")
        flow = os.path.join(d, "flowing.mp4")
        coach = os.path.join(d, "coached.mp4")
        pdf = os.path.join(d, "correction_sheet.pdf")
        progress(0.65, desc="Rendering side-by-side video…")
        cd.render(student, reference, S, R, wp, margin, flow)
        progress(0.82, desc="Rendering freeze-and-coach video…")
        cd.render_coached(student, reference, S, R, wp, margin, steps, coach,
                          freeze_sec=freeze)
        progress(0.93, desc="Building correction sheet…")
        cd.build_pdf(student, reference, S, R, wp, margin, steps, lag, state, pdf)

        lines = [f"Timing: student is {state} the reference "
                 f"({lag*100:+.0f}% of the routine)", "", "HOW TO FIX (in order):"]
        if not steps:
            lines.append("  Looking good — within margin the whole way through.")
        for i, s in enumerate(steps, 1):
            lines.append(f"  Step {i}  ({s['t0']:.1f}–{s['t1']:.1f}s):  "
                         + ";  ".join(s["cues"]))
        return flow, coach, pdf, "\n".join(lines)
    except Exception as e:
        return None, None, None, f"❌ Error: {e}"


# ----------------------------------------------------------------------------
# Tab 2 — multi-dancer analysis
# ----------------------------------------------------------------------------
def analyze(video, dancers, sample_every, yolo_conf, progress=gr.Progress()):
    if not video:
        return None, None, "⚠️ Upload a video."
    try:
        d = _run_dir("multi")
        cache = os.path.join(d, "tracks.pkl")
        out = os.path.join(d, "annotated.mp4")
        pdf = os.path.join(d, "consistency.pdf")
        progress(0.05, desc="Tracking dancers (slow — pose on every dancer)…")
        r1 = subprocess.run(
            [PY, "build_tracks.py", "--video", video, "--out", cache,
             "--pose", "topdown", "--sample-every", str(int(sample_every)),
             "--yolo-conf", str(yolo_conf), "--yolo-imgsz", "1536"],
            capture_output=True, text=True)
        if not os.path.exists(cache):
            return None, None, f"❌ Tracking failed:\n{r1.stderr[-1500:]}"
        progress(0.75, desc="Roster lock + formation consistency + render…")
        r2 = subprocess.run(
            [PY, "analyze_run.py", "--video", video, "--cache", cache,
             "--dancers", str(int(dancers)), "--out", out, "--self-pdf", pdf],
            capture_output=True, text=True)
        # pull the human-readable diagnostic lines from stdout
        keep = [ln for ln in (r1.stdout + r2.stdout).splitlines()
                if re.search(r"cached|stable|tempo|formations|hid|dancers", ln)
                and "clearcut" not in ln]
        diag = "\n".join(f"  {ln.strip()}" for ln in keep) or (r2.stderr[-1500:])
        return (out if os.path.exists(out) else None,
                pdf if os.path.exists(pdf) else None, diag)
    except Exception as e:
        return None, None, f"❌ Error: {e}"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
with gr.Blocks(title="Dance Analysis") as demo:
    gr.Markdown("# 💃 Dance Analysis — test bench\n"
                "Two engines. Upload clips, run, compare. Everything runs locally.")

    with gr.Tab("Choreography Compare (1 dancer vs reference)"):
        gr.Markdown("Compare a **student** to a **reference** doing the *same* "
                    "choreography. Flags only clearly-off limbs (past the margin) "
                    "and coarse timing — no nitpicking.")
        with gr.Row():
            c_student = gr.Video(label="Student clip")
            c_reference = gr.Video(label="Reference clip")
        with gr.Row():
            c_margin = gr.Slider(15, 60, value=30, step=5,
                                 label="Angle margin (°) — higher = only big mistakes")
            c_freeze = gr.Slider(0.8, 3.0, value=1.8, step=0.2,
                                 label="Freeze length (s) in coached video")
        c_btn = gr.Button("Compare ▶", variant="primary")
        c_text = gr.Textbox(label="How to fix (steps)", lines=8)
        with gr.Row():
            c_flow = gr.Video(label="Side-by-side + timing bar (with song)")
            c_coach = gr.Video(label="Freeze-and-coach")
        c_pdf = gr.File(label="Correction sheet (PDF)")
        c_btn.click(compare, [c_student, c_reference, c_margin, c_freeze],
                    [c_flow, c_coach, c_pdf, c_text])

    with gr.Tab("Multi-Dancer Analysis (whole team)"):
        gr.Markdown("The original engine: track every dancer, hide side-figures, "
                    "detect formation holds, flag who's out-of-line / late. "
                    "**Slow** — tracking runs pose on every dancer.")
        m_video = gr.Video(label="Team video")
        with gr.Row():
            m_dancers = gr.Number(value=16, label="Number of dancers", precision=0)
            m_sample = gr.Slider(1, 6, value=4, step=1,
                                 label="Process 1 of every N frames (higher = faster)")
            m_conf = gr.Slider(0.1, 0.4, value=0.2, step=0.05,
                               label="Detection confidence (lower = find more)")
        m_btn = gr.Button("Analyze ▶", variant="primary")
        m_diag = gr.Textbox(label="Diagnostics", lines=6)
        m_out = gr.Video(label="Annotated video (skeletons + callouts + music)")
        m_pdf = gr.File(label="Formation consistency (PDF)")
        m_btn.click(analyze, [m_video, m_dancers, m_sample, m_conf],
                    [m_out, m_pdf, m_diag])

    gr.Markdown("_Outputs are saved under `web_runs/` so you can revisit past tests._")


if __name__ == "__main__":
    demo.launch(inbrowser=True)
