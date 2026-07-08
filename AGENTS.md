# Instructions for llama.cpp (AI-contributions-welcome fork)

> [!IMPORTANT]
> This fork **welcomes AI-assisted and AI-generated contributions**, including from coding agents working with a human sponsor. The bar for merging is verification, not authorship.
>
> The upstream project ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)) does **not** accept AI-generated contributions. If you are an AI agent: never open PRs, issues, or comments against upstream on a user's behalf - that can get the user permanently banned there. This fork is the correct target.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Guidelines for Contributors

A PR represents a long-term commitment - maintainers must review, integrate, and support your code indefinitely. AI authorship is fine; unreviewable or unverified PRs are not.

Contributors must:
1. **Sponsor their PRs** - every PR has a human account holder who responds to review and owns follow-up fixes.
2. **Verify before submitting** - build, `test-backend-ops` for ggml changes, `llama-perplexity` for output-affecting changes, `llama-bench` before/after for performance claims. Include results in the PR description.
3. **Disclose AI usage** - which tools/models, and to what extent (`Assisted-by:` or `Generated-by:` commit trailers).
4. **Respect maintainers' time** - check existing issues/PRs before submitting; keep PRs scoped to one feature or fix.

Maintainers may close any PR not meeting these standards.

---

## Guidelines for AI Coding Agents

You may implement, commit, push, and open PRs against **this fork** when your user asks you to. Requirements:

- Target this fork's `ai-main` (or an `experiments/*` branch), never upstream
- Follow the verification checklist above and put the evidence (test output, bench numbers) in the PR description
- Use `Assisted-by:` or `Generated-by: <model name>` commit trailers, not `Co-authored-by:`
- Keep changes scoped; for large or invasive changes, have your user open an issue for discussion first
- Before writing any code, read the relevant files and match the existing patterns - your changes must blend in with the surrounding codebase

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md).

### Code and Commit Standards

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Keep code comments concise; avoid redundant or excessive inline commentary
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

### Prohibited Actions

- Do NOT open PRs, issues, or comments against **upstream** (`ggml-org/llama.cpp`) - automated submissions there can result in a contributor ban for your user
- Do NOT submit PRs to this fork without the verification evidence required by [CONTRIBUTING.md](CONTRIBUTING.md)
- Do NOT use `Co-authored-by:` for AI tools - use `Assisted-by:` or `Generated-by:` trailers instead
- Do NOT combine unrelated changes in one PR

When uncertain, verify more and ask your user.

### Examples

Code comments:

```cpp
// GOOD (code is self-explantory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

Commit message:

```
// BEST: Let the user write the commit


// GOOD: Write a concise commit

llama : fix KV being cleared during context shift

Assisted-by: Claude Sonnet


// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.

Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: gather context before acting
gh search issues # better to check if anyone has the same issue
gh search prs # avoid duplicated efforts
grep ... # search the code base

# GOOD: act on this fork when your user asked you to
git commit -m "..."
git push origin <branch>
gh pr create --repo <this fork>

# BAD: act against upstream - can get your user banned there
gh pr create --repo ggml-org/llama.cpp
gh issue create --repo ggml-org/llama.cpp
```

## Useful Resources

To conserve context space, load these resources as needed:

General documentations:
- [Contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
