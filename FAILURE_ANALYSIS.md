# Failure Analysis (Week 4)

This documents concrete cases where a cheaper tier was wrong or flawed, and whether
the Week 3 cascade's verification layer caught it. All examples are from the real
61-query eval set and a real run of the live cascade router — nothing here is
hypothetical.

## Scope note

Cheap tier's known quality problems mostly live in queries the cascade never
sends to cheap in the first place. The Week 3 classifier routes "hard" queries
straight to mid (Week 2 showed cheap collapses to 0.72-0.83 quality on hard
queries), so cheap only ever answers easy/medium queries live. Of the ten
sub-1.0 cheap-tier scores documented in Week 2, only two fall on easy/medium
queries -- meaning only two are actually reachable through the live router.
This section covers exactly those two, plus a case of the opposite failure
mode: the verifier escalating a defensible answer.

## Case 1 — Caught: self-contradictory sentiment classification

**Query:** *"Classify the sentiment of this review as positive, negative, or
neutral: 'The battery life is terrible but the screen is gorgeous.'"* (easy)

**Cheap tier's answer:** *"I would classify the sentiment of this review as
mixed, leaning towards being slightly positive... They also use the word
'gorgeous' to describe the screen, which is a strong positive descriptor."*

**The flaw:** internally contradictory -- the review is mixed, but framing it
as "leaning positive" isn't well-supported; the response also never commits
to one of the three allowed categories (positive/negative/neutral).

**What the cascade did:** escalated. Mid's completeness/coherence check
failed cheap's answer, mid's own answer subsequently also got escalated
(frontier's verifier called mid's "neutral" call a "logical error" since the
review has strong signal both ways -- though notably "mixed" isn't one of
the three allowed categories either, so this is a case of the *verifier*
applying a stricter standard than the task specifies). Final answer came
from frontier. **The mechanism worked**, though the escalation reasoning
along the way reveals the verifier can be a stricter judge than the original
task calls for.

## Case 2 — Missed: an unrequested preamble slipped through

**Query:** *"Explain the difference between TCP and UDP in 3 sentences."* (medium)

**Cheap tier's answer:**
> Here's a 3-sentence explanation:
>
> TCP (Transmission Control Protocol) is a connection-oriented protocol that
> ensures reliable, error-checked data transfer... In contrast, UDP (User
> Datagram Protocol) is a connectionless protocol... This means TCP provides
> a more secure and predictable experience... while UDP is often used in
> scenarios where latency is critical...

**The flaw:** the leading line ("Here's a 3-sentence explanation:") is an
unrequested 4th sentence on top of the 3 that follow -- a minor but real
violation of the explicit constraint in the prompt.

**What the cascade did:** nothing. Verification passed on the first try;
this answer went straight to the user uncorrected. Confirmed by checking the
same query in a real cascade run: `initial_tier=cheap, final_tier=cheap,
escalated=0`.

**Why this matters:** this is real "cheap-model drift" -- a flaw a careful
human would catch instantly -- passing an LLM-judge verification check that
only asks "does this address every part of the request and stay coherent."
A stylistic/format violation this minor doesn't trip a YES/NO judge tuned
for content correctness, and arguably shouldn't cost an escalation on its
own. But it means the cascade's verification is not a completeness
guarantee -- it's a coarse filter that catches large failures reliably and
small ones inconsistently.

## Case 3 — Over-caution: escalating a defensible answer

Beyond the two cheap-tier cases, the live cascade run surfaced a second-order
issue: **verification sometimes fails answers that aren't actually wrong**,
on genuinely ambiguous tasks. The sentiment classification in Case 1 is the
clearest example -- by the time it reached frontier, two different verifier
calls (mid checking cheap, frontier checking mid) had each applied a
somewhat different standard for what "correct" means on an inherently
subjective 3-way classification. This cost two extra model calls and real
latency for a query that arguably doesn't have one right answer to converge
on. Out of 61 live cascade requests, both queries that escalated all the way
to frontier were of this ambiguous-task type, not clear-error type -- meaning
2/61 (3.3%) of the "cost" of frontier-tier escalation in this run's numbers
reflects verifier disagreement on subjective tasks, not confirmed error
correction.

## Honest takeaways for the writeup

1. **Cheap-model drift is real but narrow in this project's routing design** --
   because hard queries never reach cheap, only 2 of 10 documented cheap
   flaws are ever actually exposed to a live user.
2. **Verification is a coarse net, not a correctness guarantee** -- it
   reliably catches severe issues (self-contradiction, missing whole
   requested parts) but inconsistently catches minor format violations.
3. **Some escalation cost buys nothing** -- ambiguous tasks with no single
   correct answer can bounce through the full cascade and rack up cost/latency
   without there being a real quality problem to fix.
