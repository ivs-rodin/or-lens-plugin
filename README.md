# OR Lens for Codex

OR Lens helps you understand LP and MPS optimization models: why a model is
infeasible, where coefficient ranges are wide, how rows and columns group into
families, and how HiGHS solves it. The plugin runs OR Lens and HiGHS on your
computer, so model files, matrices and solves stay local.

## Install

```sh
codex plugin marketplace add ivs-rodin/or-lens-plugin
codex plugin add or-lens@or-lens
```

After adding the marketplace you can also install OR Lens from Plugins in the
ChatGPT desktop app. Start a new chat after installing.

## Use

Ask Codex about a model file on your computer:

> Use OR Lens. Open /full/path/model.lp and explain why it is hard to solve.

The OR Lens workbench opens next to the chat with the model summary,
diagnostics, matrix, families, conflicts and runs. Select a block in the matrix
and ask about it; solves requested in the chat appear in the workbench.

Statuses, conflicts and objective values come from HiGHS and independent checks,
not from the language model. Text summaries that Codex reads become part of the
chat.

## Requirements

- macOS or Linux on arm64 or x86_64. Windows is not supported yet.
- The first start installs `uv` if it is missing and the Python dependencies
  into `~/.cache/or-lens`. It needs network access and can take a few minutes.

## Update

```sh
codex plugin marketplace upgrade or-lens
```

## License

MIT, see [LICENSE](LICENSE). Third-party components bundled into the workbench
UI are listed with their licenses in
[plugins/or-lens/THIRD_PARTY_NOTICES.md](plugins/or-lens/THIRD_PARTY_NOTICES.md).
