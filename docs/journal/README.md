# Journal

The step-by-step record of how this project was built: what was done, in what order, how
each step was checked, what broke and how it was fixed, and every judgment call. It's for
re-reading the project later, phase by phase.

## Where the record lives

| What you want | Where |
|---|---|
| Today's state: the champion, the scores, the next steps | [Context.md](../../Context.md) |
| One entry per session: Did / Learned / Scores / Next | [DIARY.md](../../DIARY.md) |
| Each phase, step by step | this folder |
| What each phase planned, task by task | [docs/superpowers/plans/](../superpowers/plans/) |
| The design, and the reasons behind big decisions | [the design](../superpowers/specs/2026-10-23-name-that-llm-design.md), [the ADRs](../adr/) |
| Each change: its diff, its original commits, its CI runs | the pull requests on GitHub, and `git log` |
| Each experiment: its run and its findings | the Kaggle notebooks (their Findings cells), and [IDEAS.md](../../IDEAS.md) |
| Every message and command, word for word | the Claude Code session transcripts (next section) |

## The full sessions

Claude Code saves each session as a transcript on the computer that ran it (under
`~/.claude/projects/`, one file per session). To reopen one, run `claude --resume` in a
terminal opened in the project folder, or use the past-conversations list of the VS Code
extension.

- Transcripts are kept for `cleanupPeriodDays` days. It was raised to 3650 (about ten
  years) on 2026-10-26; the default, 30 days, would have deleted the first sessions.
- They stay private: they never go to GitHub, because they can contain raw outputs, such
  as passages from the competition data, which the rules forbid publishing.

## Journals

- [P4: the app and continuous deployment](p4-app.md) (2026-10-26 → 27)
- [P5: fine-tuned encoders](p5-encoders.md) (2026-10-25 → 26)
- P0 to P3b: not written. Their detail is in the DIARY, the plans and the pull requests
  (#1 to #21), and the full sessions are in the transcripts.
