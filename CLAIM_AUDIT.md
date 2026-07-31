# Claim audit: language changed during archival

Every material claim changed in the archival pass, with the reason. Companion to
[`ARCHIVED_NEGATIVE_RESULT.md`](ARCHIVED_NEGATIVE_RESULT.md).

The repository was swept for the terms *Point of No Return*, *PoNR*,
*irrecoverable*, *oracle*, *ground truth*, *causal*, *exact replay*, *deterministic
replay*, *universal*, *safety guarantee*, *complete snapshot*, *validated
recovery*, *publication-ready*, and *flagship*. Most occurrences were already
correctly qualified by earlier revisions or are API and identifier names
(`ponr.py`, `point_of_no_return`, the recovery-oracle interface,
`docs/CAUSAL_ACTIONABILITY.md`'s deliberately conservative classification). Those
were retained. The table lists only what changed.

## Changed

| # | File | Claim before | Claim after | Why |
|---|---|---|---|---|
| 1 | `README.md` | Project framed as an active tool with no research verdict; verdict absent from the page | Opens with a "Research status: archived honest negative" section carrying the hypothesis, falsification, corrected result, stopping reason, and limitations | The decisive verdict must be the first thing a reader sees, not a footnote |
| 2 | `README.md` | "IPFD tells you the step at which a tested recovery controller stopped being able to save it" | "IPFD reports the step after which a tested recovery controller stopped succeeding **from restored simulator branches**", with an explicit pointer to the archived bound | The original phrasing asserts a property of the episode. The measurement is a property of restored branches, and those branches can reverse the decision |
| 3 | `README.md` | "The simulator-side recovery evidence is mid-revalidation" | "was never revalidated to the release bar, and the branch-validity study is why that work stopped rather than continued" | "Mid-revalidation" implies work in progress. No work is in progress |
| 4 | `README.md` | Evidence table had no row for restored-branch decision fidelity | Two rows added: the falsified fidelity assumption (13/120, 10/60 exact-action, 120/120 immediate equality) and the failed positive control (18/444 to 11/444, 38.9% vs 50%) | The load-bearing assumption under every other row was missing from the table |
| 5 | `README.md` | "Everything simulator-side rests on one machine and one checkpoint. Treat it as a compatibility fingerprint" | Same, plus: treat every PoNR number as a controller-relative diagnostic over restored branches, not a measurement of the uninterrupted episode | A scope caveat about hardware did not cover the validity caveat about branches |
| 6 | `README.md` | "A learned-policy headline **is promoted only when** the release evidence gate accepts a complete bundle" | "would be promoted only if... That bundle was never produced, the gate never passed, and with the project archived it is not scheduled to be" | Present tense implied a live promotion path |
| 7 | `README.md` | Compatibility section solicited discussion posts and issue reports; "Planned direction is in ROADMAP.md" | States the repository is archived and no longer solicits work; ROADMAP relabeled as historical | Soliciting contributions to an archived project misrepresents its status |
| 8 | `ROADMAP.md` | Five forward-looking directions written as plans ("The highest-value direction is...", "We want to...") | Retitled "Roadmap (historical)", opens with an archive banner and a "Why the roadmap closed" section; each direction marked dropped with its actual delivery status | Nothing on the list is planned, scheduled, or in progress |
| 9 | `docs/REVALIDATION.md` | Described what a promotion to "verified" requires, present tense, implying it is pending | Header marks the revalidation as never performed and not scheduled; adds that meeting every listed requirement would still not establish restored-branch decision fidelity | The requirements list was accurate but read as a live checklist, and it omitted the assumption that actually failed |
| 10 | `docs/RELEASE_BLOCKERS.md` | "so the next attempt does not start by tagging and hoping"; "Ordered unblock path" | Archive banner; "Ordered unblock path (not executed)"; "any future attempt" | Implied a queued release attempt |
| 11 | `CITATION.cff` | Abstract described the tool with no research status; "localizes the oracle-relative point" | Adds the archival status and the negative result; "recovery-controller-relative point... from restored simulator branches" | A citation record that omits the negative result would let the tool be cited as if validated. "Oracle-relative" also reads as if compared against a ground truth; it means relative to the supplied recovery controller |
| 12 | `EVIDENCE_LEDGER.md` | Machine-readable evidence section listed `per_branch_records.jsonl` and the three-seed traces as if both were in-repo | Records the external retention location, sizes, and digests for all large raw traces, and the archival verification result | The trace files are deliberately not in git; the ledger has to say where they are and how to check them |
| 13 | `.gitignore` | No rule for study traces | Excludes `per_branch_records.jsonl`, with a comment pointing at the manifests and the external store | 10 MB of raw per-branch records should not enter a public repository's history |

## Reviewed and retained

| Term | Where | Why it stands |
|---|---|---|
| "Point of No Return", `PoNR`, `ponr.py`, `point_of_no_return` | Throughout the library, docs, and metrics | It is the name of a defined, bounded operational quantity. The README already states it is controller-relative, that a failed recovery attempt is not proof of physical irrecoverability, and that strided probing resolves it to an interval. Change 5 adds the branch-validity bound |
| "oracle", `docs/ORACLE_CONTRACT.md`, `oracles/` | Interface name for the user-supplied recovery controller | It denotes a pluggable callable, not a source of truth. The contract document defines it explicitly |
| "irrecoverable", "irrecoverability" | `README.md`, `docs/REPRODUCE.md`, `docs/VALIDATION.md`, `docs/CAUSAL_ACTIONABILITY.md` | Every occurrence is already negated or scoped: "not proof of physical irrecoverability", "the recovery probe never established irrecoverability", or a run label |
| "causal" | `docs/CAUSAL_ACTIONABILITY.md`, `src/ipfd/actionability.py` | Names a deliberately conservative classification: did an alarm follow a known disturbance and precede the evidence-bounded boundary. The evidence ledger separately records "IPFD performs causal attribution" as **Unverified** |
| "ground truth" | `downstream_decision_results.json` field `uninterrupted_ground_truth` | The field is `null` with status `NOT_RUN_STOPPING_RULE`. It records a stage that did not run |
| "universal" | `RESEARCH_AUDIT.md`, `NOVELTY_REVIEW.md`, `HYPOTHESES.md`, `summary.json` | Used only to *reject* universality: `FALSIFIED_UNIVERSAL_DECISION_FIDELITY`, "a universal PoNR is not a novel or well-defined scientific object" |
| "complete snapshot" | `SNAPSHOT_PROTOCOLS.md`, `NOVELTY_REVIEW.md` | Used only in the negative: "Neither treatment is called a complete simulator snapshot" |
| "deterministic replay" | `NOVELTY_REVIEW.md` rejected-ideas list | Listed as an idea that was rejected: "Causal onset from deterministic replay. Invalid without an explicit intervention, matched exogenous variables, and causal estimand" |
| "flagship" | `EVIDENCE_LEDGER.md`, `README.md`, `ARCHIVED_NEGATIVE_RESULT.md` | Every occurrence is a denial. The ledger row "Counterfactual Branch Validity should continue as the flagship direction" is marked **Falsified**; the other two state that IPFD did not demonstrate a flagship research capability |
| "safety guarantee" | Nearest is `RESEARCH_AUDIT.md`: "Backend independence, real-time operation, formal safety, or sim-to-real validity" | That row is marked **Unsupported** with the note "These claims must not be made" |
| "publication-ready", "exact replay", "validated recovery" | (no occurrences outside this audit) | Nothing to change |

## Not changed, deliberately

`RESEARCH_AUDIT.md`, `NOVELTY_REVIEW.md`, `HYPOTHESES.md`, `EXPERIMENT_PROTOCOL.md`,
`CORRECTED_EXPERIMENT_PROTOCOL.md`, and `SNAPSHOT_PROTOCOLS.md` were written during
the study and already state the negative outcome, the stopping decision, and the
limitations accurately. Editing them after the fact would degrade the evidence
trail. `EVIDENCE_LEDGER.md` received only the additions in row 12; no existing row
was reworded.

No result file, provenance record, or manifest was modified. The three-seed and
five-seed cohorts are byte-identical to what the study produced.
