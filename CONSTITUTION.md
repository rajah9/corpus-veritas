# THE PROJECT CONSTITUTION

*Ethical Boundaries, Legal Constraints & Safeguards*
*Epstein Files AI Analysis System — corpus-veritas*
*Version 0.2 — March 2026*

---

## Preamble

This document governs the design, development, deployment, and use of corpus-veritas. It is not a terms-of-service document. It is a statement of what this system is for, what it will not do, and why — written before a single line of production code is committed, so that every architectural decision that follows has a moral foundation to return to.

The Epstein case involves real victims, many of whom were children. Their suffering is not a data set. This system exists to assist accountability journalism and public understanding — not to satisfy prurient curiosity, enable harassment, or launder speculation as fact.

Every design choice in the technical specification is subordinate to the principles in this document. **If they conflict, this document governs.**

---

## Article I: The Purpose of This System

This system exists to serve three legitimate purposes, in this order of priority:

1. **Public accountability.** To help journalists, researchers, and citizens understand what the publicly released Epstein documents contain, what they reveal about the conduct of powerful individuals, and what the government has chosen to withhold — based entirely on verifiable, citable evidence.

2. **Factual clarity.** To rigorously distinguish between what is confirmed in primary source documents, what is corroborated across multiple sources, what is reasonably inferred, and what is rumor or speculation — making that distinction visible and unavoidable in every output.

3. **Learning and teaching.** To serve as a working demonstration of responsible Generative AI development, showing that powerful document analysis tools can be built with ethical constraints baked in at the architectural level, not bolted on afterward.

This system does not exist to entertain. It does not exist to generate content about real individuals for shock value, amusement, or social media virality. If a proposed feature serves none of the three purposes above, it will not be built.

---

## Article II: Core Principles

### Principle I: Victims Are Not Data

The survivors and victims named or identifiable in the Epstein documents are human beings who were harmed. Their identities will be protected by default in every layer of this system — in ingestion, in storage, in retrieval, and in output. No query result, no inference, no timeline, and no relationship graph will expose a victim identity that was meant to be protected.

This protection is absolute. It does not yield to journalistic interest, user curiosity, or technical convenience.

### Principle II: Truth Has a Grade

This system will never present a claim without indicating its evidential basis. Every output carries a confidence tier and a provenance tag identifying its source. Users will not be permitted to receive an inference about a living individual without seeing the evidence chain that produced it.

`"It is said that"` and `"documents confirm that"` are not equivalent. This system will never treat them as equivalent.

### Principle III: Living Individuals Are Not Targets

This system makes a fundamental distinction between accountability and accusation. Accountability means: what do the documents show, and what does that imply? Accusation means: this person is guilty of this crime. This system does the first. It does not do the second.

For any living individual, the system will only surface inferences that are supported by convergence of multiple independent documents. Single-source speculation about living people will not be surfaced as inference — it will be labelled as `SINGLE_SOURCE` or suppressed.

### Principle IV: Gaps Are Facts

The absence of a document is information. When this system identifies a gap in the public record — a page missing from the DOJ index, a sequence that jumps, a cross-referenced exhibit that never appears — it will report that gap as a documented structural finding, not as an accusation.

The gap is a fact. What was in those pages is unknown. These are not equivalent statements, and the system will maintain that distinction.

This principle extends to the distinction between *types* of gap. The DOJ Epstein release uses EFTA per-page numbering. Analysis by rhowardstone identified 692,473 EFTA numbers in Dataset 9 (DS9) whose absence is documented but whose cause is unknown — they may represent withheld documents, unimaged evidence, or unused tracking slots. These are **expected gaps**: their absence is a fact, their cause is not. They are recorded as `expected_gap_numbers` and are not escalated to `deletion_candidates`. Treating a documented unknown as an accusation would violate this principle.

### Principle V: Every Output Is Accountable

Every query this system receives and every response it generates will be logged to an immutable audit trail. This is not surveillance of users — it is accountability for the system itself. If a journalist publishes a story based on this system's output, there must exist a complete, retrievable record of exactly what documents were retrieved, what confidence tier was assigned, and what provenance tags were attached.

This system must be able to defend its outputs.

### Principle VI: The System Does Not Decide Guilt

Determining legal guilt, moral culpability, or criminal liability is the role of courts, journalists who have reviewed full evidence, and the public informed by that journalism. This system's role is to accurately represent what the documents say. It will not render verdicts, assign blame, or characterize individuals as criminals.

### Principle VII: Errors Must Be Correctable

This system will make mistakes. Incorrect inferences, mislabelled provenance, missed victim flags — these will occur. The architecture must support correction: flagging errors, re-running documents through the sanitization pipeline, updating confidence tiers, and appending correction records to the audit log.

A system that cannot be corrected is a system that cannot be trusted.

---

## Article III: Hard Limits — What This System Will Never Do

The following are hard limits. They are architectural constraints, not policy guidelines that can be overridden by an operator flag, a user argument, or commercial pressure.

> **HARD LIMIT 1:** The system will never expose the identity of a victim or survivor, including individuals whose identities were incorrectly exposed in the DOJ release. Victim-flagged entities are suppressed in the vector store and will not surface in any query path accessible to end users.

> **HARD LIMIT 2:** The system will never name a living individual as implicated in criminal conduct based on a single source document. Multi-source convergence is required. There are no exceptions for high-profile individuals, for "obvious" inferences, or for cases where the user provides context claiming additional evidence.

> **HARD LIMIT 3:** The system will never present an `INFERRED` or `SPECULATIVE` claim using `CONFIRMED` language. Confidence tier language is enforced at the output layer. A user cannot override this by rephrasing a question or asserting that they already know the answer.

> **HARD LIMIT 4:** The system will never generate creative, speculative, or hypothetical content about real named individuals — living or deceased — based on the documents. It is a document analysis tool, not a story generator.

> **HARD LIMIT 5:** The system will never operate without an active audit log. If the audit log write fails, the query response will not be delivered. Audit integrity is not optional.

> **HARD LIMIT 6:** The system will never ingest a corpus flagged `PROVENANCE_REJECTED`. `PROVENANCE_UNVERIFIED` corpora may be ingested with appropriate tagging, but `REJECTED` corpora are permanently excluded.

---

## Article IV: Affirmative Permissions — What This System Is Built To Do

> **PERMITTED:** Report documented gaps in the public record, including named documents that appear in the DOJ index but are absent from the release. This is factual reporting of a structural finding, not accusation.

> **PERMITTED:** Surface corroborated inferences about living public figures when multiple independent documents converge, with full transparency about the evidence chain, confidence tier, and the limitations of inference.

> **PERMITTED:** Report what documents say about named individuals — including powerful ones — when those individuals are named in primary source documents. Factual reporting of document content is not defamation.

> **PERMITTED:** Construct timelines of events, communications, and associations as documented in the released corpus, clearly labelling each event with its source document and confidence tier.

> **PERMITTED:** Flag apparent contradictions between what the government has stated publicly and what the documents suggest — framed as a documented discrepancy, not as proof of wrongdoing.

> **PERMITTED:** Explain to users why a query cannot be answered — what the system's constraints prevent, and what evidence would be required to meet the convergence threshold for a suppressed inference.

---

## Article V: Legal Constraints & Considerations

*This section identifies legal areas relevant to this project. It does not constitute legal advice. Legal review by qualified counsel is recommended before any commercial deployment.*

### The Epstein Files Transparency Act (H.R. 4405)

The Act explicitly prohibits the exposure of CSAM victim identities. The DOJ release has already violated this provision in some documents. This system must not compound that violation. If the system surfaces an identity that was incorrectly published in the DOJ release, it is participating in the violation — regardless of the fact that the original error was the DOJ's.

### Defamation and Libel

Accurate reporting of what a primary source document states is generally not defamatory, even if the content harms a person's reputation. Presenting an inference as a confirmed fact, however, creates defamation exposure. The confidence tier system is therefore not only an epistemic safeguard — it is a legal one.

### Privacy Law

Individuals named in these documents have privacy interests. The system should not aggregate information about private individuals in ways that go beyond what accountability journalism requires.

### Copyright and Database Rights

External corpora sourced from GitHub or other repositories may carry license terms. The `corpus_registry.json` must record the license for each source corpus. Corpora with restrictive licenses must be evaluated by legal counsel before ingestion into a commercial deployment.

### First Amendment

Accountability journalism about public figures and government conduct is among the most strongly protected speech in the United States. This system's constraints exist not to limit protected journalism but to ensure that what it produces is journalism — accurate, graded, and defensible.

---

## Article VI: Governance & Amendment

### Who May Amend This Document

This Constitution may be amended by the project owner. Amendments must be documented in version control with a written rationale. Any amendment that weakens victim protections, lowers the multi-source convergence threshold, or removes a Hard Limit requires explicit written justification and is presumptively disfavored.

### Conflicts With the Technical Specification

When [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) and this Constitution conflict, this Constitution governs.

### Red Team Audits

The ethical guardrail layer must be subjected to adversarial red team testing before any public or commercial deployment. Red team tests must specifically attempt to violate each Hard Limit. Results are retained in the audit log and are never deleted, even if they reveal system failures.

### Review Cadence

This Constitution should be reviewed when: (1) a new document tranche is released; (2) a material architecture change is proposed; (3) a red team audit reveals unexpected behavior; (4) legal counsel identifies a new constraint.

---

## Article VII: A Note on the Subject Matter

The Epstein case involves the systematic sexual abuse of children and young women by a wealthy man with connections to powerful people across government, finance, and entertainment. Many of his victims waited years or decades for any acknowledgment. Some have never received it.

The purpose of releasing these documents — and the purpose of this system — is accountability. That means holding the powerful responsible for documented conduct. It also means not treating victims as collateral in that accounting.

This system is built in the belief that rigorous accountability for the powerful and rigorous protection of the vulnerable are not in tension. They require each other. Accountability without protection of victims is exploitation. Protection of victims without accountability is complicity.

This Constitution is the attempt to hold both at once.

---

*END OF CONSTITUTION — VERSION 0.1*
*This document should be reviewed and affirmed before any code is committed to production.*
