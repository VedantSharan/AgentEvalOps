# Round-2 Annotation — Step-by-Step Instructions for Each Annotator

Follow these steps **exactly, in order**. Each annotator works **alone, on their own computer, in their own Label Studio**.
Expected time: setup 15–20 min, labelling 1–2 hours (40 trajectories).

| Annotator ID | Who | Export file to produce |
|---|---|---|
| **A1** | the person who was A1 in round 1 | `supporting_files/labels/round2/annotator1_raw.json` |
| **A2** | the person who was A2 in round 1 | `supporting_files/labels/round2/annotator2_raw.json` |

---

## Part 1 — Get the latest repository

Open a terminal (Linux/macOS) or **PowerShell** (Windows) **inside your existing `AgentEvalOps` folder**.

```bash
git checkout main
git pull
```

Check you have the round-2 files:

```bash
git log --oneline -3
```
You should see a commit mentioning *round-2*. The folder must contain:
- `data/labels/round2_sample.json`
- `code/annotation/label_studio_config_v2.xml`
- `docs/annotation_guidelines_v2.md`

If `git pull` says *"Your local changes would be overwritten"*, stop and tell the team (do not delete anything).

---

## Part 2 — Python environment (skip if you already have a working `.venv`)

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
(If PowerShell refuses to run the activate script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then activate again.)

Your prompt must now start with `(.venv)`. Record your versions (send these lines to the team):
```bash
python --version
python -m pip show label-studio
```

---

## Part 3 — Verify the sample is the right one

```bash
python -m pytest tests/test_sample_round2.py
```
Must print `2 passed`. Then check the file fingerprint:

- Linux: `md5sum data/labels/round2_sample.json`
- macOS: `md5 data/labels/round2_sample.json`
- Windows: `Get-FileHash data\labels\round2_sample.json -Algorithm MD5`

Must be **`693f59f7be42c57890855bc7999e1acc`** (Windows shows it in capitals — that is fine).
If it differs, **stop** and tell the team: you would be labelling a different sample.

---

## Part 4 — Create your Label Studio project

1. Start Label Studio (keep this terminal open while you work):
   ```bash
   label-studio start
   ```
   A browser tab opens at `http://localhost:8080`. Log in with your own account.
2. **Create Project** → Project Name: `AgentEvalOps Round 2 - A1` (or `- A2`).
3. Tab **Labeling Setup** → **Custom template** → switch to **Code** view → delete everything in the box → open `code/annotation/label_studio_config_v2.xml` in a text editor → copy **all** of it → paste.
4. **Test the form in the preview on the right before saving:**
   - Click `success` → the *First Error Step* and *Note* fields must **disappear**.
   - Click `failure` → they must **appear**.
   - If the preview shows a red error, stop and send a screenshot to the team.
5. Click **Save**.
6. **Import** → upload `data/labels/round2_sample.json` → **Import**. You must see **40 tasks**.
7. Project **Settings** → **Annotation** (older versions: **General**) → *Task Sampling* → choose **Sequential sampling** → **Save**.
   This makes everyone see the items in the same order (`presentation_order` 1 → 40).
8. Do **not** import any other file, and do **not** enable predictions / ML backend.

---

## Part 5 — Label (the important part)

Before the first item, read **all of `docs/annotation_guidelines_v2.md`** (about 10 minutes). Keep it open beside Label Studio.

Then click **Label All Tasks** and, for every item:

1. Read the prompt, the gold answer, the agent's final answer, and **the whole transcript**.
2. Fill **in this order**, without going back:
   1. **Outcome** — `success` / `failure` / `error`
   2. **Primary failure category** — `no_failure` if success, otherwise the *earliest* error (guidelines §3)
   3. **First error step** (failures only) — the first step that leaves a valid solution path (guidelines §4)
   4. **Note** (failures only) — one line: what went wrong, at which step
3. Click **Submit**.

Rules:
- Work alone. Do not discuss items, and **do not open the other annotator's export** (even if it appears after a `git pull`) until your own export is saved.
- Do not look at grader outputs (`data/processed/regrade_v2.csv`, `supporting_files/…`) while labelling.
- Take breaks if needed; Label Studio saves each submitted item. Do not rush — time per item is recorded and reported.
- If an item is genuinely unclear, choose your best label and explain in the note. Do not skip items.

---

## Part 6 — Export and hand in

> **A1:** the commands below are written for A2. Everywhere you see `annotator2_raw.json` or `(A2)`, use `annotator1_raw.json` and `(A1)` instead.

1. In the project, click **Export** → choose **JSON** (not *JSON-MIN*, not CSV) → **Export**. A file like `project-12-at-2026-10-22-….json` downloads.
2. Create the folder and copy the file **without opening or editing it**:

   **Linux / macOS**
   ```bash
   mkdir -p supporting_files/labels/round2
   cp ~/Downloads/project-*-at-*.json supporting_files/labels/round2/annotator2_raw.json   # A1 uses annotator1_raw.json
   ```
   **Windows (PowerShell)**
   ```powershell
   mkdir supporting_files\labels\round2 -Force
   Copy-Item $HOME\Downloads\project-*-at-*.json supporting_files\labels\round2\annotator2_raw.json   # A1 uses annotator1_raw.json
   ```
   (If you have several `project-*` files in Downloads, type the exact file name instead of the `*` pattern.)
3. Quick check that it is a complete export (must print `40 40`):
   ```bash
   python -c "import json;d=json.load(open('supporting_files/labels/round2/annotator2_raw.json'));print(len(d),sum(len(t['annotations'])==1 for t in d))"
   ```
4. Commit **only your file** and push:
   ```bash
   git add supporting_files/labels/round2/annotator2_raw.json
   git commit -m "Part C: round-2 raw Label Studio export (A2)"
   git pull --rebase
   git push
   ```
5. Send the team: the commit hash (`git log --oneline -1`), your Python and Label Studio versions, the date you labelled, and roughly how long it took.

Done. Thank you!
