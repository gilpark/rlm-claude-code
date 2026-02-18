# SPEC-16: Epistemic Verification (Hallucination Detection)

## Overview

This specification defines an always-on epistemic verification system that detects procedural hallucinations—instances where Claude has the right information but fails to use it correctly or makes claims not supported by cited evidence.

**Inspired by**: [Pythea/Strawberry](https://github.com/leochlon/pythea) hallucination detection toolkit.

## Problem Statement

LLMs exhibit "procedural hallucinations" where they:
- Retrieve correct information but fail to use it (e.g., counting letters correctly, then stating wrong count)
- Cite evidence that doesn't actually support their claims
- Present confident answers disconnected from provided context
- Confabulate details that sound plausible but aren't grounded in evidence

**Key insight**: The issue isn't that models lack knowledge—they "know but don't use" it correctly.

## Design Constraints

### Claude API Limitation

Unlike OpenAI, Anthropic's API **does not expose token logprobs**. This prevents direct implementation of Strawberry's "evidence scrubbing + confidence shift" methodology:

```
# Strawberry approach (requires logprobs):
p1 = P(correct | full context)      # via logprobs
p0 = P(correct | scrubbed context)  # via logprobs
gap = KL(p1 || p0)                  # information-theoretic
```

### Alternative: Consistency-Based Verification

For Claude, we use **semantic consistency checking**:

```
# Claude-compatible approach:
answer_with = claude(question, evidence)     # Full context
answer_without = claude(question, no_evidence)  # Scrubbed
similarity = semantic_compare(answer_with, answer_without)

if similarity > threshold:
    # Answer didn't change → evidence wasn't actually used
    flag_as_potential_hallucination()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EPISTEMIC VERIFICATION                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  CLAIM EXTRACTOR                      │   │
│  │  • Parse response into atomic claims                  │   │
│  │  • Identify cited evidence spans                     │   │
│  │  • Map claim → evidence dependencies                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 EVIDENCE AUDITOR                      │   │
│  │  • Verify each claim against its cited evidence      │   │
│  │  • Detect unsupported extrapolations                 │   │
│  │  • Flag phantom citations                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               CONSISTENCY CHECKER                     │   │
│  │  • Scrub evidence and re-query                       │   │
│  │  • Compare semantic similarity                       │   │
│  │  • Compute "evidence dependence score"               │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              VERIFICATION RESULT                      │   │
│  │  • Per-claim confidence scores                       │   │
│  │  • Evidence gaps (bits equivalent)                   │   │
│  │  • Flagged claims with reasons                       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. REPL Functions (Explicit Verification)

New helper functions available in the REPL sandbox:

```python
# Verify a single claim against evidence
verify_claim(
    claim: str,
    evidence: str | list[str],
    threshold: float = 0.95
) -> ClaimVerification

# Audit a chain of reasoning steps
audit_reasoning(
    steps: list[dict],  # [{claim: str, cites: list[str]}]
    sources: dict[str, str]  # {span_id: content}
) -> list[ClaimVerification]

# Compute evidence dependence score
evidence_dependence(
    question: str,
    answer: str,
    evidence: str
) -> float  # 0.0 = answer independent, 1.0 = fully dependent

# Auto-detect and verify claims in a response
detect_hallucinations(
    response: str,
    context: dict[str, str]
) -> HallucinationReport
```

### 2. Reasoning Traces Integration

Extend `ReasoningTraces` with epistemic tracking:

```python
# New node types
VALID_DECISION_TYPES = frozenset({
    "goal", "decision", "option", "action", "outcome", "observation",
    "claim",       # NEW: Atomic claim with cited evidence
    "verification" # NEW: Verification result for a claim
})

# New edge labels
EDGE_LABELS = {
    "cites",      # claim → evidence_source
    "verifies",   # verification → claim
    "refutes",    # verification → claim (negative)
}

# API extensions
class ReasoningTraces:
    def add_claim(
        self,
        decision_id: str,
        claim: str,
        evidence_ids: list[str],
        confidence: float = 0.5
    ) -> str:
        """Add a claim with cited evidence to a decision."""

    def verify_claim(
        self,
        claim_id: str,
        verification_model: str = "haiku"
    ) -> VerificationResult:
        """Verify a claim against its cited evidence."""

    def get_epistemic_gaps(
        self,
        goal_id: str
    ) -> list[EpistemicGap]:
        """Get all claims with insufficient evidence support."""

    def get_verification_report(
        self,
        goal_id: str
    ) -> EpistemicReport:
        """Full verification report for a goal tree."""
```

### 3. Orchestrator Integration (Always-On)

Verification checkpoint in the main orchestrator loop:

```python
class RLMOrchestrator:
    async def process(self, query: str) -> str:
        # ... existing processing ...

        # NEW: Epistemic verification checkpoint
        if self.config.verification.enabled:
            report = await self._verify_response(response, context)

            if report.has_critical_gaps:
                if self.config.verification.on_failure == "retry":
                    response = await self._retry_with_evidence_focus(
                        query, report.flagged_claims
                    )
                elif self.config.verification.on_failure == "flag":
                    response = self._annotate_uncertain_claims(
                        response, report
                    )

        return response
```

### Smart Verification Skip (Performance Optimization)

To reduce latency and API costs, verification is **automatically skipped** for simple responses:

```python
# Skip verification if:
# 1. No final answer
# 2. Verification disabled
# 3. Response is very short (< 100 chars) - likely simple answer
# 4. No evidence files to verify against
should_verify = (
    state.final_answer
    and self.verification_config.enabled
    and len(state.final_answer) >= 100
    and len(context.files) > 0
)
```

**Rationale**: Simple queries like "say hello" or "what is 2+2" don't need epistemic verification. This optimization:
- Eliminates unnecessary 2nd LLM call for simple responses
- Reduces latency by ~50% for straightforward queries
- Saves API costs on trivial interactions

## Data Types

### ClaimVerification

```python
@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""

    claim_id: str
    claim_text: str
    evidence_ids: list[str]

    # Verification scores
    evidence_support: float      # 0.0-1.0, how well evidence supports claim
    evidence_dependence: float   # 0.0-1.0, how much answer changed without evidence
    consistency_score: float     # 0.0-1.0, semantic consistency across variations

    # Computed metrics
    confidence_justified: bool   # Is the claimed confidence supported?
    evidence_gap_bits: float     # Information gap (Strawberry-compatible)

    # Flags
    is_flagged: bool
    flag_reason: str | None      # "unsupported", "phantom_citation", etc.
```

### EpistemicGap

```python
@dataclass
class EpistemicGap:
    """An identified gap between claim and evidence."""

    claim_id: str
    claim_text: str
    gap_type: Literal[
        "unsupported",           # No evidence supports claim
        "partial_support",       # Evidence supports part of claim
        "phantom_citation",      # Cited source doesn't exist
        "contradicted",          # Evidence contradicts claim
        "over_extrapolation",    # Claim goes beyond evidence
        "evidence_independent",  # Answer unchanged without evidence
    ]
    gap_bits: float              # Information gap in bits
    suggested_action: str        # What to do about it
```

### HallucinationReport

```python
@dataclass
class HallucinationReport:
    """Full hallucination detection report."""

    response_id: str
    total_claims: int
    verified_claims: int
    flagged_claims: int

    claims: list[ClaimVerification]
    gaps: list[EpistemicGap]

    # Summary metrics
    overall_confidence: float    # Weighted average claim confidence
    max_gap_bits: float          # Largest evidence gap
    has_critical_gaps: bool      # Any gaps > threshold

    # Recommendations
    should_retry: bool
    retry_guidance: str | None
```

## Verification Methods

### Method 1: Direct Evidence Check

Ask Claude directly if the claim is supported:

```python
async def verify_claim_direct(
    claim: str,
    evidence: str,
    model: str = "haiku"
) -> float:
    """Direct verification via LLM."""

    prompt = f"""
    Evaluate whether this claim is fully supported by the evidence.

    CLAIM: {claim}

    EVIDENCE:
    {evidence}

    Rate support from 0.0 (completely unsupported) to 1.0 (fully supported).
    Consider:
    - Does the evidence explicitly state what the claim asserts?
    - Is the claim an extrapolation beyond the evidence?
    - Are there any contradictions?

    Respond with just a number and brief reason.
    """

    response = await claude.complete(prompt, model=model)
    return parse_score(response)
```

### Method 2: Evidence Scrubbing (Consistency Check)

Compare answers with and without evidence:

```python
async def verify_claim_scrubbing(
    question: str,
    claim: str,
    evidence: str,
    model: str = "haiku"
) -> float:
    """Consistency-based verification via evidence scrubbing."""

    # Get answer WITH evidence
    answer_with = await claude.complete(
        f"Given this evidence: {evidence}\n\nAnswer: {question}",
        model=model
    )

    # Get answer WITHOUT evidence
    answer_without = await claude.complete(
        f"Answer this question: {question}",
        model=model
    )

    # Compare semantic similarity
    similarity = semantic_similarity(answer_with, answer_without)

    # High similarity = evidence wasn't actually used
    evidence_dependence = 1.0 - similarity

    return evidence_dependence
```

### Method 3: Claim Decomposition

Break complex claims into verifiable atomic statements:

```python
async def decompose_and_verify(
    response: str,
    context: dict[str, str],
    model: str = "haiku"
) -> list[ClaimVerification]:
    """Decompose response into claims and verify each."""

    # Extract atomic claims
    claims = await extract_claims(response, model=model)

    # Identify cited evidence for each claim
    claims_with_evidence = await map_claims_to_evidence(
        claims, context, model=model
    )

    # Verify each claim
    results = []
    for claim, evidence_ids in claims_with_evidence:
        evidence = "\n".join(context[eid] for eid in evidence_ids)

        support = await verify_claim_direct(claim, evidence, model)
        dependence = await verify_claim_scrubbing(
            claim, claim, evidence, model
        )

        results.append(ClaimVerification(
            claim_text=claim,
            evidence_ids=evidence_ids,
            evidence_support=support,
            evidence_dependence=dependence,
            # ... compute other fields
        ))

    return results
```

## Configuration

### Config Schema

```python
@dataclass
class VerificationConfig:
    """Epistemic verification configuration."""

    # Enable/disable
    enabled: bool = True

    # Verification thresholds
    support_threshold: float = 0.7      # Min evidence support
    dependence_threshold: float = 0.3   # Min evidence dependence
    gap_threshold_bits: float = 2.0     # Max acceptable gap

    # Behavior on failure
    on_failure: Literal["flag", "retry", "ask"] = "flag"
    max_retries: int = 2

    # Performance
    verification_model: str = "haiku"   # Model for verification
    max_claims_per_response: int = 10   # Limit for cost control
    parallel_verification: bool = True  # Verify claims in parallel

    # Modes
    mode: Literal["full", "sample", "critical_only"] = "sample"
    sample_rate: float = 0.3            # For sample mode
```

### Integration with Existing Config

```python
@dataclass
class RLMConfig:
    # ... existing fields ...

    # NEW: Epistemic verification
    verification: VerificationConfig = field(
        default_factory=VerificationConfig
    )
```

### Bypass Mechanisms

Verification can be bypassed via:
- `/simple` command (existing)
- `verification.enabled = false` in config
- Per-query: `llm(query, verify=False)` in REPL

## Implementation Phases

### Phase 1: Core Types & REPL Functions

**SPEC-16.01-16.10**: Foundation

| ID | Requirement |
|----|-------------|
| 16.01 | Define ClaimVerification, EpistemicGap, HallucinationReport types |
| 16.02 | Implement `verify_claim()` REPL function |
| 16.03 | Implement `audit_reasoning()` REPL function |
| 16.04 | Implement `evidence_dependence()` REPL function |
| 16.05 | Implement `detect_hallucinations()` REPL function |
| 16.06 | Add claim extraction via Claude |
| 16.07 | Add evidence mapping via Claude |
| 16.08 | Add semantic similarity comparison |
| 16.09 | Unit tests for all REPL functions |
| 16.10 | Integration tests with real Claude API |

### Phase 2: Reasoning Traces Integration

**SPEC-16.11-16.20**: Persistence & Tracking

| ID | Requirement |
|----|-------------|
| 16.11 | Add "claim" decision type to ReasoningTraces |
| 16.12 | Add "verification" decision type |
| 16.13 | Add "cites" edge label |
| 16.14 | Add "verifies"/"refutes" edge labels |
| 16.15 | Implement `add_claim()` method |
| 16.16 | Implement `verify_claim()` method (traces version) |
| 16.17 | Implement `get_epistemic_gaps()` method |
| 16.18 | Implement `get_verification_report()` method |
| 16.19 | Schema migration for new tables |
| 16.20 | Tests for traces integration |

### Phase 3: Orchestrator Integration

**SPEC-16.21-16.30**: Always-On Verification

| ID | Requirement |
|----|-------------|
| 16.21 | Add VerificationConfig to RLMConfig |
| 16.22 | Implement verification checkpoint in orchestrator |
| 16.23 | Implement claim flagging on failure |
| 16.24 | Implement retry with evidence focus |
| 16.25 | Add verification to trajectory output |
| 16.26 | Implement parallel claim verification |
| 16.27 | Add cost tracking for verification calls |
| 16.28 | Implement sample mode for cost control |
| 16.29 | Tests for orchestrator integration |
| 16.30 | End-to-end integration tests |

### Phase 4: Polish & Optimization

**SPEC-16.31-16.40**: Production Readiness

| ID | Requirement |
|----|-------------|
| 16.31 | Implement critical_only mode |
| 16.32 | Add verification caching |
| 16.33 | Optimize prompt templates |
| 16.34 | Add rich output for verification results |
| 16.35 | Implement `/verify` slash command |
| 16.36 | Add verification to trajectory analysis |
| 16.37 | Property tests for verification logic |
| 16.38 | Security tests for claim extraction |
| 16.39 | Performance benchmarks |
| 16.40 | Documentation and examples |

## File Structure

```
src/
├── epistemic/                    # NEW: Epistemic verification module
│   ├── __init__.py
│   ├── types.py                 # ClaimVerification, EpistemicGap, etc.
│   ├── claim_extractor.py       # Parse responses into claims
│   ├── evidence_auditor.py      # Verify claims against evidence
│   ├── consistency_checker.py   # Evidence scrubbing verification
│   ├── verification_engine.py   # Main verification orchestration
│   └── prompts.py               # Verification prompt templates
├── repl_environment.py          # Add new REPL functions
├── reasoning_traces.py          # Extend with claim/verification types
└── orchestrator/
    └── core.py                  # Add verification checkpoint
```

## Cost Analysis

### Per-Response Costs (Haiku verification)

| Operation | Input Tokens | Output Tokens | Cost |
|-----------|--------------|---------------|------|
| Claim extraction | ~500 | ~200 | ~$0.0003 |
| Evidence mapping | ~300 | ~100 | ~$0.0002 |
| Per-claim verification | ~200 | ~50 | ~$0.0001 |
| Total (5 claims) | ~1300 | ~500 | ~$0.0008 |

### Cost Mitigation

1. **Sample mode**: Only verify 30% of claims randomly
2. **Critical only**: Only verify claims with uncertainty markers
3. **Caching**: Cache verification results for identical claim/evidence pairs
4. **Haiku default**: Use cheapest model for verification

## Success Criteria

[SPEC-16.SC01] Verification SHALL detect >80% of procedural hallucinations in test suite.

[SPEC-16.SC02] Verification SHALL add <500ms latency in sample mode.

[SPEC-16.SC03] Verification cost SHALL be <10% of base response cost.

[SPEC-16.SC04] False positive rate SHALL be <5% (claims incorrectly flagged).

[SPEC-16.SC05] System SHALL gracefully degrade when verification is disabled.

## Open Questions

1. **Semantic similarity implementation**: Use embedding distance? LLM-as-judge? Both?

2. **Claim extraction reliability**: How to handle ambiguous claim boundaries?

3. **Evidence span identification**: How to map claims to specific evidence sections?

4. **Confidence calibration**: How to convert consistency scores to "bits" equivalent?

5. **User feedback loop**: Should users be able to mark false positives/negatives?

## References

- [Pythea/Strawberry](https://github.com/leochlon/pythea) - Original hallucination detection toolkit
- [Semantic Entropy (Nature 2024)](https://www.nature.com/articles/s41586-024-07421-0) - Entropy-based confabulation detection
- [HaluGate (vLLM 2025)](https://blog.vllm.ai/2025/12/14/halugate.html) - Token-level hallucination detection
- [MetaQA (arXiv 2025)](https://arxiv.org/abs/2502.15844) - Metamorphic relation-based detection

## Changelog

- **2026-01-15**: Initial specification created
