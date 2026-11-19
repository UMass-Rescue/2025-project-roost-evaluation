### Description

#### Purpose and change

Closes #N. Explain the problem, who it affects, and the resulting behavior in a
short high-level narrative. Connect the change to the linked issue. Keep this
aligned with the README; explain implementation details only when they help a
reviewer assess the change.

#### How to try it

- **Setup:** Prerequisites, configuration, and exact setup/start commands (link to the root [README.md](../README.md) Quick Start if unchanged).
- **Exercise the change:** Exact commands or steps, sample inputs, and expected results. For a bug fix, include the original failure and how to verify it is fixed.
- **Interactive artifacts:** Applicable example scripts, sample data, or demos.
- **Happy-path smoke test:** The command, what it covers, and the expected output (for example `make smoke-test` or `make video-smoke-test`).
- **Cleanup:** How to stop services and remove only the test resources.

#### Documentation and figures

List README, figure, and other documentation updates with links, or explain why
no updates were needed. Confirm that affected examples and commands are current.

### Checklist

- [ ] The project launches successfully
- [ ] I have created tests which fail without the change (if possible)
- [ ] All new and existing tests are passing
- [ ] All pipelines are passing (including style)
- [ ] I have added docstrings to functions that warrant them
- [ ] I have commented especially unclear sections of the code
- [ ] I have performed a self-review of this change
- [ ] I have extended the README / other external documentation, where necessary
