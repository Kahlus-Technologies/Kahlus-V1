# Fetch list — for Zotero

**There is no single domain.** These are eight papers in eight different venues.
Nothing to scrape; use identifiers.

## Fastest path

Zotero → **Add Item by Identifier** (the magic-wand icon), paste one per line:

```
10.5664/jcsm.9538
10.1111/j.1365-2869.2008.00700.x
10.3390/s22083079
10.1038/s41592-024-02362-y
```

Those four DOIs are **from model memory, not verified against the publisher.** If
one fails to resolve, that is the expected failure mode and it means the DOI is
wrong, not that the paper doesn't exist — fall back to searching the title.

The other four have no DOI I can vouch for. Search by title:

| Search this title | Venue | Note |
| --- | --- | --- |
| `SSF-SET: A Discrete EEG Token-based Framework for Sleep Stage Forecasting` | IEEE J Biomed Health Inform, 2026 | Needs Stanford access |
| `Variability and predictability in human sleep` | Brain Communications, 2025 | Open access |
| `Del Pup EEG cross-validation leakage` | arXiv / PubMed | Title uncertain |
| `Apicella EEG data leakage` | arXiv / PubMed | Title uncertain |

`references.bib` in this folder has full entries and can be imported directly —
but read its provenance header first, because it marks which fields are verified
and which are not.

## Verified permalinks

These resolve to the actual papers and were returned by a literature search, so
they are the most reliable pointers here:

- Lee 2022 — https://consensus.app/papers/details/10e1400a92775d99926f6aea3219f891/
- Danker-Hopfe 2009 — https://consensus.app/papers/details/8be56fd4cf6b5c69a83c42bd241ce19b/
- Kweon 2026 — https://consensus.app/papers/details/c2aa1a05230a5658959edeeabb953ca6/
- Leguia 2025 — https://consensus.app/papers/details/f7ece62572ba5aaea011bf5d35313d8b/
- Hussain 2022 — https://consensus.app/papers/details/a56569cda5185ce9b901dabec2090550/

No such link exists for Bernett, Del Pup, or Apicella — those three were never
returned by a search in this project. They come from an earlier brief and have
**never been verified.**

## Priority

If you only fetch two: **Lee 2022** and **Leguia 2025**.

Lee is the κ table the entire §3.3 argument rests on. Leguia is the AUC 0.88 number
you will be asked about.

## What to check while reading

**Bernett, Del Pup, Apicella.** The handbook claims all three address leakage
*between* training examples and none addresses leakage *inside* a single example.
That claim is the whole novelty argument for F9/F10. It has never been verified by
reading them. If any one of them does cover within-example overlap, the novelty
claim needs rewording — and it is much better that you find that than a reviewer.
