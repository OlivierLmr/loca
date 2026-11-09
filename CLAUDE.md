# loca

## Version control / deployment
- Repo: `~/code/iict/loca`, public at https://github.com/OlivierLmr/loca.
- Commit directly on `main`; push after each commit without asking.
- Commits are GPG-signed (`commit.gpgsign = true`); keep it that way.
- "Deploy" = `cargo install --path .` (replaces `~/.cargo/bin/loca`). Never restart a running
  loca yourself — the user restarts it when convenient.
