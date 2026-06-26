"""Consolidator matching-quality evaluation harness.

Measures the entity-resolution matcher's precision and recall against
**real ground truth** (no large committed corpus):

  * precision  — LEI homonyms: one (name_clean, country) mapping to >1
                 distinct LEI is >1 real company the matcher must NOT fuse.
                 Computed live against the graph (see live_report).
  * recall     — VAT-variants (same VAT, different name) found in the graph,
                 plus a *generated* perturbation set (clean-invariant noise
                 must still match). The generator is code; nothing is stored.

`clean()` is a validated pure-Python replica of apoc.text.clean (100% on
399 diverse prod names) so the recall eval runs in CI without Neo4j.
"""
