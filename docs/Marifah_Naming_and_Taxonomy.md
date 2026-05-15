# Marifah — Naming and Taxonomy

**Date:** 2026-05-13
**Status:** Working document. Captures the naming thought process for the company, mechanisms, architecture family, and the taxonomic vocabulary used to position competing systems. Not a positioning artifact — internal source-of-truth that downstream artifacts (deck, website, paper, blog post) draw from.

---

## §1 — The thesis the vocabulary is built around

Knowledge is recognition, not retrieval.

Two traditions converged on this from opposite ends of intellectual history:

- *In the West:* Helmholtz's 1867 perception-as-inference, through Rao and Ballard's predictive coding, Hinton's Helmholtz Machine, and Friston's free energy principle.
- *In the East:* al-Ghazali through Suhrawardi, Ibn ʿArabi, and Mulla Sadra — eight centuries of Sufi epistemology distinguishing *maʿrifa* (recognition-based knowing) from *ʿilm* (propositional knowledge).

These traditions weren't asking the same question. The Western lineage was working out a theory of perception and brain function. The Eastern lineage was working out a theory of how the soul comes to know reality. They arrived at structurally parallel insights through entirely independent investigation.

The convergence is the foundation of the vocabulary. The names below aren't decorative — each one is anchored in one of these two traditions, and the layered structure of names reflects the layered structure of the architecture itself.

**Note on chronology:** the Eastern lineage is best stated in chronological order as al-Ghazali (d. 1111) → Suhrawardi (d. 1191) → Ibn ʿArabi (d. 1240) → Mulla Sadra (d. 1640). A thematic ordering (unity metaphysics → illuminationism → synthesis) places Ibn ʿArabi before Suhrawardi, which is sometimes useful but reads as chronologically incorrect to readers fluent in Islamic philosophy. Default to chronological in external writing unless context warrants the thematic order.

---

## §2 — The vocabulary

### 2.1 Marifah — the company

Anglicized from Arabic *maʿrifa* (معرفة) — recognition, gnosis, direct experiential knowledge. Distinguished in Sufi epistemology from *ʿilm* (acquired propositional knowledge) and *kashf* (unveiling). The most general term in the tradition for the kind of knowing that arises through recognition rather than acquisition.

Marifah is both the company name and the name of its defining mechanism (Recognition Cortex). When the context is ambiguous: capitalize "Marifah" for the company, lowercase or italicize *marifa* / *maʿrifa* for the mechanism or the underlying concept.

**Romanization commitment:** "Marifah" with terminal H for English stability. The *ʿ* diacritic is reserved for academic and long-form writing (e.g., "ʿIlm-class") where precision matters more than friction.

### 2.2 Nous — the Reasoning Cortex substrate

From Greek νοῦς (nous) — intellect, intuitive intellect, the faculty of direct intellectual apprehension. In Plotinus, the second hypostasis (between the One and Soul). In Aristotle, the active intellect that makes thought possible. In Islamic philosophy, rendered as *ʿaql* (عقل), with elaborate hierarchies (al-Farabi, Avicenna).

Nous names the reasoning substrate — the layer that does deliberate, iterative, multi-step inference. It's the layer where predictive coding, ACT, hierarchical abstraction, and constraint propagation live. The Reasoning Cortex.

**Why Greek and not Arabic ʿaql:** the Greek-Arabic synthesis isn't an awkward graft — Islamic philosophy translated *nous* into *ʿaql* and built on it for centuries. Using *nous* keeps the Western connection explicit while *marifa* and *ʿilm* anchor the Eastern. Using *ʿaql* would erase the Western thread. Both terms are correct in their respective traditions; we use the one that signals the synthesis.

### 2.3 CORAL — the first architecture in the family

**C**ortical **R**easoning via **A**bstraction **L**ayers.

CORAL is the first instance of the architecture family. It implements the Nous substrate (predictive coding, ACT, hierarchical abstraction, constraint propagation) and the Marifah mechanism (Recognition Cortex / Hierarchical Multi-Scale Codebook / crystallization-as-recognition).

Future architectures in the family — CORAL v2, or differently-named successors — extend or revise these mechanisms. The architecture *name* is for the specific implementation; the *category name* (CRA) is for the family.

**Acronym note:** the previous expansion was different; the renamed expansion ("via Abstraction Layers") reflects that hierarchical abstraction is one of CORAL's defining mechanisms and is structurally accurate to the architecture as built.

### 2.4 CRA — Cortical Reasoning Architecture (the category)

The category that CORAL is the first instance of. A CRA is any architecture that:

1. Implements deliberate reasoning at a Nous-class substrate
2. Implements recognition-based compounding at a Marifah-class mechanism
3. Is structured around cortically-inspired mechanisms (predictive coding, hierarchical abstraction, crystallization, precision gating)

CRA sits *alongside* Connectionism, Symbolic AI, and Neuro-Symbolic AI as a peer category — not inside any of them. This positioning is essential: the temptation will be for analysts and competitors to file CRA inside Neuro-Symbolic AI ("Type 5 with emergent symbols" or similar), which collapses the category claim. Hold the boundary: *Cortical Reasoning Architectures are a distinct fourth category.*

### 2.5 CWA — Cognitive Workflow Automation (the customer-facing product category)

The buyer-facing category. What an enterprise procures. The analyst-relations term. The thing on a budget line.

CWA distinguishes from RPA (mechanical, no reasoning), IPA (saturated analyst term meaning RPA + some ML, doesn't claim reasoning), LLM agents (probabilistic, unreliable), Decision Intelligence (Aera — decision recommendation, not workflow execution), and Process Intelligence (Celonis — process mining, not runtime reasoning).

CWA is the *product category*. CRA is the *architectural category*. They're two layers serving two audiences. Customers procure CWA; the architecture that makes their CWA work is a CRA; the first CRA in commercial production is CORAL; the company that produces it is Marifah.

### 2.6 The taxonomic vocabulary — ʿIlm-class / Nous-class / Marifah-class

The single most strategically important piece of the naming system. The taxonomy that places every AI reasoning system in one of three classes, each named for the kind of knowing it implements.

**ʿIlm-class systems.** From Arabic *ʿilm* (علم) — acquired propositional knowledge. Knowledge as stored facts and patterns retrievable from memory. ʿIlm-class systems store and recombine patterns from training data. Their knowledge is what they were given.

LLMs are the canonical ʿIlm-class systems. They are extraordinary at ʿIlm-class problems (recall, recombination, surface-pattern matching, language generation). They are structurally limited on tasks that require Nous-class or Marifah-class capabilities (faithful multi-step reasoning, recognition-based compounding, deterministic execution).

This is not a pejorative classification. ʿIlm is one of the three forms of knowledge in the source tradition. Some problems are ʿIlm-class problems and ʿIlm-class systems are the right tools. The taxonomic claim is about *what kind of system* a given architecture is, not about quality.

**Nous-class systems.** From Greek *νοῦς* — intellect, deliberate inference. Nous-class systems perform iterative reasoning, propagate state through structured representations, and reach conclusions through computation rather than retrieval.

Reasoning architectures (HRM, TRM, GRAM, the Reasoning Cortex substrate of CORAL) are Nous-class. They do something LLMs structurally cannot: faithful multi-step deliberation that doesn't drift. They lack what Marifah-class systems have: recognition-based compounding that gets cheaper on familiar patterns.

**Marifah-class systems.** From Arabic *maʿrifa* — recognition, direct experiential knowing. Marifah-class systems compound at deployment time: encountering a pattern repeatedly causes the system to recognize that pattern faster, with less computation, on future encounters. Knowledge is built up not by retrieval (ʿIlm) or by step-by-step inference (Nous) but by recognition that accumulates with experience.

CORAL with the Recognition Cortex is the first commercial Marifah-class system. Verses AI's Genius is the closest research-stage neighbor (Friston-inspired, active inference). No Marifah-class system has been deployed at enterprise scale.

**Why this taxonomy is strategically durable:**

If LLMs improve dramatically at reasoning — as they're doing — they remain ʿIlm-class systems. Better ʿIlm-class systems, but ʿIlm-class. The class distinction doesn't dissolve with scaling. This is what makes the taxonomy a more durable strategic frame than "LLM-alternative." The latter erodes as LLMs improve; the former doesn't.

**Why the hierarchy in the source tradition matters:**

In Sufi epistemology, the three forms of knowing are not equal. *ʿIlm* is the most accessible and the most easily acquired. *Maʿrifa* is the rarest and the most direct. The taxonomy carries this hierarchy with it — implying that Marifah-class systems are doing something epistemically more sophisticated than ʿIlm-class systems.

This is provocative. It's also defensible from the source tradition and from the technical architecture. The provocation is the point — it forces the conversation onto our ground.

---

## §3 — How the vocabulary nests

```
Marifah (company)
   │
   ├── operates in the category: CWA (Cognitive Workflow Automation)
   │
   ├── builds CRA (Cortical Reasoning Architecture) products
   │       │
   │       └── first product/architecture: CORAL
   │               │
   │               ├── Nous substrate (Reasoning Cortex)
   │               │      └── PC, ACT, hierarchical abstraction, constraint propagation
   │               │
   │               └── Marifah mechanism (Recognition Cortex)
   │                      └── HMSC codebook, crystallization, compounding-at-deployment
   │
   └── taxonomic positioning vocabulary: ʿIlm-class / Nous-class / Marifah-class
           │
           ├── ʿIlm-class: LLMs and retrieval/pattern-matching systems
           ├── Nous-class: reasoning architectures (HRM, TRM, CORAL substrate)
           └── Marifah-class: recognition-based compounding systems (CORAL with Recognition Cortex)
```

The naming layers serve different audiences and contexts:

- **Customer / enterprise buyer:** hears CWA and Marifah. Doesn't need to learn the rest.
- **Analyst / industry observer:** hears CWA, Marifah, and the taxonomy. Needs the taxonomy to compare us with competitors.
- **Investor (sophisticated):** hears the full stack. The depth signals seriousness.
- **Academic / paper reader:** hears CRA, CORAL, Nous, Marifah, full taxonomic discussion. Lineage and rigor matter most here.
- **Talent / advisor:** hears the full stack, philosophically. The vocabulary is part of why they want to be associated.

---

## §4 — Operational rules for using the vocabulary

### 4.1 Capitalization and italicization

- **Marifah** — capitalized when referring to the company.
- *marifa* / *maʿrifa* — lowercase italics when referring to the Sufi concept or the underlying mechanism (when context makes "Marifah the company" ambiguous).
- **Nous** — capitalized when referring to the Reasoning Cortex substrate. Lowercase *nous* in italics when discussing the Greek philosophical concept.
- **CORAL** — all caps (it's an acronym).
- **CRA / CWA** — all caps.
- **ʿIlm-class / Nous-class / Marifah-class** — capitalized as taxonomic categories. Lowercase italics for the underlying concepts (*ʿilm*, *nous*, *maʿrifa*).

### 4.2 Diacritics

- The *ʿ* (ʿayn) diacritic appears in *ʿIlm* and *ʿIlm-class* — preserve it in academic, long-form, and paper contexts.
- Do not attempt to use diacritics in URLs, file names, code identifiers, or branding assets (logos, decks). "Ilm-class" without diacritic is acceptable shorthand for these contexts.
- "Marifah" never uses a diacritic in English; the original *maʿrifa* does in Arabic, but the Anglicized form is the company name.

### 4.3 Pluralization

- *Marifah-class systems* (not "Marifahs").
- *Nous-class architectures* (not "Nouses").
- *ʿIlm-class models* (not "ʿIlms").
- The class adjective makes the plural work without forcing the underlying terms into awkward English pluralization.

### 4.4 What to use in which context

| Context | Lead vocabulary |
|---|---|
| Customer demo / sales meeting | CWA, Marifah, simple English ("Recognition Cortex," "Reasoning Cortex") |
| Investor pitch (broad) | CWA, CRA, Marifah, brief mention of taxonomy |
| Investor pitch (technical / sophisticated) | Full stack including taxonomy |
| Analyst briefing | CWA + taxonomy. The taxonomy is what they'll cite. |
| Academic paper | CRA, CORAL, Nous, Marifah, taxonomic justification, source tradition citations |
| Blog post (general audience) | Marifah, taxonomy, simple English. Brief philosophical depth. |
| Blog post (philosophical / thought leadership) | Full vocabulary, source traditions, the convergence thesis |
| Talent recruitment | Full stack, philosophy front and center |
| Patent filings | CRA, CORAL, technical mechanism names. Avoid religiously-coded terms in claims; keep them in background sections only. |

### 4.5 What to avoid

- Mixing Romanizations of the same term within one document. Pick "Marifah" and stick to it.
- Using "Marifah-class" and "Marifah" in the same sentence without disambiguation. "Marifah's marifa mechanism" — works. "Marifah is a Marifah-class system" — works but verbose; consider "Marifah is the first commercial Marifah-class system" instead.
- Claiming that Sufis or Greek philosophers anticipated modern cognitive science. They didn't. The convergence thesis is about parallel structural insights through independent investigation, not about historical anticipation.
- Treating the taxonomy as a pejorative on ʿIlm-class systems. It's a structural classification, not a judgment of merit. LLMs are extraordinary at ʿIlm-class problems. The taxonomy makes claims about *kind*, not *quality*.

---

## §5 — Naming decisions resolved

| Decision | Resolution | Rationale |
|---|---|---|
| Company name | Marifah | Anchors the intellectual identity in the Sufi tradition; matches the defining mechanism (Recognition Cortex) |
| Defining mechanism name | Marifah (= Recognition Cortex) | Company and mechanism share the name deliberately |
| Substrate name | Nous (= Reasoning Cortex) | Greek tradition for deliberate reasoning; pairs structurally with Marifah/Eastern tradition for recognition |
| First architecture | CORAL — Cortical Reasoning via Abstraction Layers | Preserves CORAL identity; renames acronym to accurately reflect hierarchical abstraction mechanism |
| Architecture category | CRA — Cortical Reasoning Architecture | Unchanged from prior strategic work |
| Customer-facing category | CWA — Cognitive Workflow Automation | Buyer-facing; what gets procured |
| Taxonomic classes | ʿIlm-class / Nous-class / Marifah-class | Three classes anchored in the three modes of knowing in the source tradition |
| Repo name | marifah-core | Aligns with company name |

---

## §6 — Outstanding operational tasks

These follow from adopting the vocabulary; they don't change the vocabulary itself.

1. **Trademark search on Marifah** — multiple jurisdictions, both Marifah and Marifa spellings.
2. **Domain registration** — marifah.ai, marifah.com (and variants) before any public use of the name.
3. **SEO landscape audit** — Marifah currently has Islamic religious content presence; understand what we're competing against and decide whether to fight for the term or accept second-page ranking initially.
4. **Existing branding wind-down** — Aktuator references in any prior decks, drafts, public communications, GitHub bios, etc. Coordinate the rename so the transition is clean rather than messy.
5. **Game plan revision** — update game plan §1 (positioning section) to reflect the full vocabulary. Refactor §1.3 competitive map using ʿIlm-class / Nous-class / Marifah-class as the organizing axis.
6. **Repo creation** — `marifah-core` per build plan §3 Week 2.
7. **Chronology fix in any prior writing** — Suhrawardi (d. 1191) precedes Ibn ʿArabi (d. 1240). Any text listing them in the other order needs correction before external use.

---

## §7 — Document maintenance

This is a living vocabulary document. Updates as:

- Naming decisions evolve (e.g., a new mechanism gets added to the architecture family — name it within this vocabulary)
- New competitive positioning emerges (e.g., a competitor announces something that needs to be placed in the taxonomy)
- Source-tradition references deepen (e.g., we engage more rigorously with a specific Sufi thinker and want to reflect that in positioning)
- External writing draws from this doc and surfaces needed clarifications

Versioned alongside the game plan; if positioning shifts substantively, both docs move together.

---

*End of vocabulary document. Status: active reference. All future positioning writing draws from this.*
