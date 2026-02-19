# Externalizing Causation: Toward AI That Evolves With Its Environment

*February 2026*

---

## The Wrong Question

The RLM paper asks: how do we process larger contexts more effectively?

This is the wrong question.

Context size is an engineering problem. It will be solved — repeatedly — by hardware, by better models, by improved tooling. In the 1990s, RAM was a constant constraint. Developers wrote elaborate schemes to fit programs into 640KB. Then RAM grew. The schemes became unnecessary. The problem dissolved.

Context windows are today's RAM. They will grow. The question of how to handle 500,000 tokens will become the question of how to handle 5,000,000, then 50,000,000. The techniques built to work around today's limits will become legacy code.

The right question is different:

> **Can an AI's work product continuously evolve with its environment?**

This is not a hardware problem. It is an architecture problem. And it does not dissolve as context windows grow.

---

## What "Evolving With the Environment" Means

Consider what a skilled human collaborator does over time on a codebase.

They remember not just what they concluded, but why. When you mention that the auth service is behaving strangely, they don't start from zero. They recall that JWT expiry was flagged last month, that a refresh token problem was identified but deprioritized, that the auth module has a known dependency on the cache layer.

More importantly: they notice when their understanding needs updating. When a file they reasoned about changes, they flag the conclusions that depended on it. When an assumption turns out to be wrong, they revise conclusions built on that assumption.

They are, in a precise sense, **causally aware**. They track not just knowledge, but the chains that produced it — and they know which chains are sensitive to which changes.

Current AI systems have none of this. Every session starts cold. Prior conclusions cannot be trusted because there is no way to know if the premises that produced them still hold. The AI cannot say "I analyzed this before, but auth.py changed since then, so my prior analysis may be invalid." It can only look again from scratch, as if it had never looked at all.

This is the problem that matters. Not token count.

---

## Externalization as a Principle

The key move in Zhang et al. (2025) is not a technique for handling long prompts. It is a principle:

> **What the model cannot hold in memory, it should externalize — and then navigate actively.**

Zhang applied this to context. The model's working context was too large to reason about coherently in one pass, so it was moved outside the model's context window into a REPL. The model no longer receives context passively — it navigates it. It peeks, searches, and loads only what each sub-task requires.

The REPL is not the insight. Externalization is.

This reframes the question: not "how do we handle large context?" but "what else should we externalize, and how?"

We apply the same principle to the temporal dimension. A model's reasoning history — the causal chain of how it came to know what it knows — cannot survive a session boundary. It is lost on reset. So we externalize it: move it into a persistent causal store the model can navigate in future sessions.

| | Zhang (2025) | This work |
|---|---|---|
| **What is externalized** | Context (spatial) | Causation (temporal) |
| **Mechanism** | REPL | REPL + hooks + frame store |
| **Scope** | Within session | Across sessions |
| **Model's role** | Navigate context actively | Navigate reasoning history actively |

This is not an ontological shift from Zhang's approach. It is the same principle, extended. The model navigates its environment — spatially in REPL, temporally in the causal store. The mechanism is the same. The target is different.

---

## What to Externalize — And Why Causation

There are three obvious candidates.

**Inputs and outputs**: save what the model saw and what it said. This produces logs. Logs become stale. There is no mechanism to detect when a logged conclusion is no longer valid. This is state storage. It captures what was true at a moment, not how truth was established. When the environment changes, there is no way to know which logged conclusions are now suspect.

**REPL state**: save the execution environment — variables, intermediate results, code. This fails because the REPL is a computational artifact, not a reasoning artifact. It records what the model computed, not why. And more fundamentally: the LLM's context window resets regardless. A saved REPL state injected into a new session is data without provenance. The model cannot reason about whether to trust it.

**Causation**: save the chain of reasoning — what was observed, what was inferred, what was assumed, and under what conditions each conclusion would no longer hold.

Causation is the right unit because it is the only one that supports three necessary operations:

*Verification*: can we re-trace the chain and confirm the conclusion still holds?

*Invalidation*: when a premise changes, which conclusions depended on it?

*Navigation*: given a new question, which prior reasoning is relevant and still valid?

Logs support none of these. REPL state supports none of these. A causal store supports all three.

---

## The Structure: Call Trees, Not Logs

RLM execution is recursive — this is not an implementation detail, it determines the correct storage structure.

When a model decomposes "find the auth problem" into "analyze logs," "trace code," and "check dependencies," these are not independent queries. They are children of a parent frame. Their conclusions flow upward. Their errors propagate upward.

```
root: "find auth problem"
├── child: "analyze prod logs"
│     evidence: "401 patterns at /api/auth"
│     invalidated if: log format changes
├── child: "trace error to source"
│     evidence: "auth.py line 47, no expiry check"
│     invalidated if: auth.py modified
└── root conclusion: aggregated from children
      invalidated if: either child invalidated
```

A flat log of these results loses the dependency structure. When `auth.py` changes, there is no way to know which downstream conclusions were built on analysis of that file — unless the storage structure preserves the causal chain.

The call tree mirrors execution structure naturally. Storage that fights this shape adds complexity without adding information.

The unit of storage is a **CausalFrame**:

```
query:                  what was asked
context_slice:          what this frame was allowed to see
evidence:               which frames and observations it relied on
conclusion:             what it concluded
confidence:             how certain
invalidation_condition: what would make this wrong
status:                 active | suspended | completed | invalidated | promoted
branched_from:          if this was a pivot, which frame it diverged from
```

The `invalidation_condition` field is not optional. It is the mechanism by which the system avoids confident wrongness. Every conclusion carries its own kill switch.

---

## Invalidation as Evolution

The mechanism by which AI work products evolve with their environment is invalidation.

When `auth.py` changes:
- Find all frames where `auth.py` was in the context slice
- Mark those frames invalidated
- Cascade to all frames that depended on their conclusions
- Surface: "these conclusions may no longer hold"

When a new session begins:
- Compare file hashes to prior session
- Propagate invalidation for changed files
- The model begins not from zero, but from a map of what it knows, what has changed, and what needs re-examination

This is not a search problem. It is a tree traversal on 10-20 nodes — trivial to compute, significant in consequence.

---

## Branches Are Part of the Record

Human reasoning is not linear. When a line of inquiry is abandoned — "let's set aside the JWT angle and look at the DB layer" — the abandoned work should not be deleted. It should be suspended.

The reasons it was abandoned, the conclusions it reached before abandonment, and the condition under which it might be worth resuming are all information. A suspended branch often becomes relevant again: the DB hypothesis fails, and the JWT angle is worth revisiting with fresh evidence. A system that deleted the prior work forces re-exploration from scratch. A system that suspended it enables resumption.

Two fields are sufficient: `status` and `branched_from`. The tree already encodes the branching structure.

---

## The Vision

With causal externalization, a different kind of interaction becomes possible.

A developer says: "auth 쪽 어떻게 됐어?"

The AI does not search logs or reprocess transcripts. It navigates its causal store:
- Prior analysis: JWT expiry issue, confidence 0.9, auth.py line 47
- auth.py was modified yesterday — that analysis is now invalidated
- DB hypothesis was explored and suspended — still resumable
- Related documentation has not been updated since the invalidation

The AI responds with a current picture, surfaces what has changed, and — without being asked — flags the documentation that now needs updating because the code it described has changed.

This is not large context processing. This is **causal awareness**: knowing what you know, how you came to know it, and whether it is still true.

The user speaks contextually. The AI understands immediately. Related work updates without being asked. Not because the context window is large enough to hold everything — but because the reasoning structure makes the right connections automatically.

---

## Scale and Simplicity

A temptation when designing this system is to reach for complexity: DAG structures, vector databases, ML routing.

The scale does not justify any of that.

A focused reasoning session produces 10-20 frames at depth 2-3. At that scale, O(n) linear scan is instant. A flat dict fits in memory trivially. JSONL per session requires zero dependencies and is human-readable.

Every piece of complexity added beyond what scale demands will be debugged, maintained, and eventually removed. The right design solves the actual problem at the actual scale.

---

## A Note on This Paper

This paper was produced through a process that mirrors what it describes.

The ideas here did not arrive fully formed. They emerged through a session of questions, pivots, and refinements — each tracked as a chain of reasoning, not a flat list of conclusions:

- An early design stored state (flat CausalTrace). Invalidated when we recognized execution is a tree.
- A complex MemoryStore with SQLite, tiers, and vector search was proposed. Suspended when scale constraints made it unnecessary.
- The framing of "how to handle large context" was the starting point. Pivoted when the RAM analogy revealed it as the wrong question.

Each of those shifts has an implicit `invalidation_condition`. The scale assumption can be revisited if frame counts grow. The JSONL approach if query patterns require indexing. The two-field branch management if more complex patterns emerge.

This is not a coincidence. The problem of AI reasoning across sessions is the problem of any cognitive system that needs to function over time: what to keep, what to discard, and how to know which conclusions remain valid.

We built the answer by doing the thing the answer describes.

---

## Summary

Context windows will keep growing, as RAM did in the 1990s. The constraint will keep moving.

The underlying question — can this system remain coherent as its environment evolves? — will not change.

Externalizing causation is an answer to that question. Not because it solves the token problem, but because it gives the model the structure it needs to know what it knows, why it knows it, and when it should no longer trust what it knows.

The model should navigate its environment, not be overwhelmed by it. Spatially, that means REPL. Temporally, that means causal frames.

> *The question is not how much the model can hold.*
> *The question is whether what the model knows can evolve.*

---

*Based on: Zhang et al., "Recursive Language Models" (2025)*
*Companion implementation: rlm-claude-code*
*February 2026*