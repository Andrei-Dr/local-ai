# Migration: flatten repo bench/box/* -> bench/*, then make /ai on the box a git checkout

**Decided 2026-09-20 (Andrei):** rearrange the REPO to match the box's flat runtime layout, not the
other way. Box layout (/ai/bench/*.sh + *.py + preflight.sh, co-located, relative ./qualbench.sh
calls, queue.tsv -> /ai/bench/*.sh) is the runtime truth; repo's bench/box/ split is artificial.

## DO NOT run while either is live
- Box queue running (kq1->ctx1->mainqual->h2qual->orf1->r1b). Waiter b659kukf9 fires on drain.
- Qwen worktree active on brief 16 (paths bench/box/slotlib.py, ctxproxy.py). Waiter b83oqxg20.
Execute only when BOTH have fired and are merged/idle.

## Steps (one atomic changeover, all repo-side except the final box git init)
1. Repo: `git mv bench/box/<f> bench/<f>` for all 148 files (no basename collisions — verified).
2. Repo: sed the ~20 path refs `bench/box/` -> `bench/` in: bench/jobreport.py, SPEC.md, notes.md,
   research/*-report.md, research/qwen-queue/*.md + reports/*.md. Grep after to confirm zero `bench/box/` left.
3. Repo: run the test suite (bench/tests) green; `bash -n` a couple box scripts.
4. Commit: `refactor: flatten bench/box -> bench to mirror the box /ai/bench runtime layout 1:1`.
5. Box: `cd /ai && git init && git remote add origin git@github.com:Andrei-Dr/local-ai.git`.
   `.gitignore`: models/ src/ .venv/ *.gguf *.log logs/ traces/ runs/ slots/ *.slot server_*.log build_*.log *.bak-*
   `git fetch && git checkout -f main -- bench/` (or reset paths so /ai/bench == repo bench/).
6. Box: verify `git status` shows only real edits (no 91GB of models). Verify a dry job still runs
   (paths unchanged on the box: /ai/bench/*.sh stays flat — this migration makes the REPO match it,
   so NOTHING on the box moves; box identity already set: Andrei-Dr <66381299+Andrei-Dr@users.noreply.github.com>, SSH auth OK).
7. Going forward: author box scripts in the repo, `git pull` on the box to deploy; box can commit results up.

## Why nothing on the box breaks
The box already runs flat. Flattening the repo makes repo == box. queue.tsv, relative calls, preflight
sourcing all keep working because the box filesystem is untouched — only the repo is rearranged to match.
