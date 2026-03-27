#!/usr/bin/env python3
"""Write updated case study findings and supporting article files."""

import os

ARTICLE_DIR = "/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article"

def write_findings():
    content = """# WikiArt Case Study — Findings (Updated for New Pipeline)

## Dataset
- **Source**: Kaggle `steubk/wikiart` (81,446 total images)
- **Sample**: 49,585 images (proportional 50K sample, seed=42)
- **Styles**: 27 art-historical categories
- **Artists**: 1,092 unique
- **Processing**: ImageSpace CPU-only (ONNX Runtime on Apple M-series)

## Processing Performance (MEASURED)
- **Atlas generation**: 49,585 images in 432.5s (7.2 min), 115 img/s
- **CLIP embeddings**: ONNX CPU — 1084.0s (18.1 min), 45.7 img/s
- **PCA**: 512d -> 50d in <0.1s (69% variance retained)
- **openTSNE (FFT)**: 26.2s
- **HDBSCAN**: 19.0s -> 19 clusters discovered
- **k-NN (k=10, cosine)**: 15.5s
- **Cluster labels (CLIP)**: 1.3s
- **Metadata extraction**: 418.7s (7.0 min)
- **Image features**: 423.1s (7.1 min)
- **Total**: 2441.6s (40.7 min) CPU-only

## Clustering: HDBSCAN (19 clusters, density-based)

Unlike the previous KMeans (K=15) clustering, HDBSCAN discovers the number of clusters
automatically from the density structure of the t-SNE-projected embeddings. Noise points
are reassigned to the nearest cluster via cKDTree distance. Each cluster receives a
CLIP-generated semantic label.

## Style Distribution (Top 10)
| Style | Count | % |
|---|---|---|
| Impressionism | 7,939 | 16.0% |
| Realism | 6,579 | 13.3% |
| Romanticism | 4,281 | 8.6% |
| Expressionism | 3,953 | 8.0% |
| Post Impressionism | 3,938 | 7.9% |
| Symbolism | 2,753 | 5.6% |
| Baroque | 2,638 | 5.3% |
| Art Nouveau Modern | 2,583 | 5.2% |
| Abstract Expressionism | 1,620 | 3.3% |
| Northern Renaissance | 1,594 | 3.2% |

## Cluster Analysis (19 HDBSCAN Clusters)

### Cluster Sizes and Labels
| Cluster | CLIP Label | n |
|---|---|---|
| 0 | Complex Scenes - Vast Landscapes | 1,197 |
| 1 | Complex Scenes - Battle Scenes | 2,035 |
| 2 | Art Nouveau | 1,631 |
| 3 | Baroque | 6,903 |
| 4 | Portraits - Sketches | 3,998 |
| 5 | Vast Landscapes - Romantic Landscapes | 2,241 |
| 6 | Vast Landscapes - Mountain Landscapes | 2,617 |
| 7 | Portraits - Warm-Toned | 5,244 |
| 8 | Portraits - Expressionist | 1,881 |
| 9 | Portraits - Close-up Portraits | 1,837 |
| 10 | Portraits - Baroque | 738 |
| 11 | Romantic Landscapes | 886 |
| 12 | Still Life - Interiors | 1,129 |
| 13 | Still Life - Impressionist | 1,308 |
| 14 | Impressionist | 889 |
| 15 | Expressionist | 1,219 |
| 16 | Abstract Expressionist | 7,185 |
| 17 | Seascapes | 1,435 |
| 18 | Vast Landscapes - Impressionist | 5,212 |

### Key Finding 1: CLIP Groups by Visual/Formal Similarity, Not Period

Most clusters are mixed. The largest cluster (Cluster 3, n=6,903, labeled "Baroque")
combines: Baroque (19.4%), Northern Renaissance (11.4%), Romanticism (10.9%),
Early Renaissance (10.8%). Four centuries of figurative art grouped by shared formal
properties: figural composition, dark-ground portraiture, oil technique.

This confirms Rashtchian et al.'s (2023) finding that embeddings encode "substance"
over "style."

### Key Finding 2: CLIP Isolates Ukiyo-e (Bias Evidence)

Ukiyo-e distribution:
- Cluster 0: 709/726 (97.7% of all Ukiyo-e)
- All other clusters combined: 17 images (2.3%)

Ukiyo-e makes up 59.2% of Cluster 0 (the rest is Art Nouveau Modern 15.4%,
Naive Art 4.3%, Post-Impressionism 3.8%).

No Western style achieves comparable concentration in any single cluster.
The highest Western purity is Art Nouveau Modern at 56.1% (Cluster 2) and Cubism at
53.1% (Cluster 15). But Ukiyo-e's 97.7% assignment to a single cluster is vastly
higher than any Western style's concentration.

**Compared to old finding**: Previously 86% of one cluster was Ukiyo-e. Now Ukiyo-e
is 59.2% of Cluster 0, but 97.7% of all Ukiyo-e goes to that cluster. The bias
finding is even stronger: HDBSCAN, which discovers clusters from density, still
isolates Ukiyo-e almost entirely.

### Key Finding 3: Abstract Art Forms Coherent Family

Cluster 16 (n=7,185, labeled "Abstract Expressionist"):
- Abstract Expressionism: 1,555 (21.6%)
- Color Field Painting: 974 (13.6%)
- Expressionism: 886 (12.3%)
- Minimalism: 826 (11.5%)

These four styles account for 59.0% of the cluster. CLIP treats mid-to-late
twentieth-century abstraction as variations on a shared formal vocabulary.

### Key Finding 4: HDBSCAN Discovers Granular Landscape Taxonomy

Unlike KMeans, HDBSCAN distinguished FIVE distinct landscape clusters:
- Cluster 5: Vast Landscapes - Romantic Landscapes (n=2,241)
- Cluster 6: Vast Landscapes - Mountain Landscapes (n=2,617)
- Cluster 11: Romantic Landscapes (n=886)
- Cluster 17: Seascapes (n=1,435)
- Cluster 18: Vast Landscapes - Impressionist (n=5,212)

Each has a distinct CLIP-generated label. This granularity reveals sub-genres
within landscape painting that a coarser clustering would obscure.

### Key Finding 5: Cubism Forms Its Own Cluster

Cluster 15 (n=1,219): Cubism 53.1%, Expressionism 15.5%, Synthetic Cubism 8.9%.
The density-based approach detected Cubism as a distinct stylistic island rather
than merging it with other form-distortion movements.

### Key Finding 6: Portrait Sub-genres

HDBSCAN discovered FIVE portrait-related clusters:
- Cluster 4: Portraits - Sketches (n=3,998)
- Cluster 7: Portraits - Warm-Toned (n=5,244)
- Cluster 8: Portraits - Expressionist (n=1,881)
- Cluster 9: Portraits - Close-up Portraits (n=1,837)
- Cluster 10: Portraits - Baroque (n=738)

This granularity demonstrates that CLIP distinguishes not just "portraits" as a
category but sub-traditions within portraiture (sketched vs. close-up vs. Baroque
vs. Expressionist vs. warm-toned).

## Cluster Purity Rankings
| Cluster | Dominant Style | % |
|---|---|---|
| 0 | Ukiyo-e | 59.2% (but 97.7% of Ukiyo-e goes here) |
| 2 | Art Nouveau Modern | 56.1% |
| 15 | Cubism | 53.1% |
| 14 | Impressionism | 51.9% |
| 18 | Impressionism | 47.1% |
| 8 | Expressionism | 46.2% |
| 17 | Impressionism | 40.1% |
| 10 | Romanticism | 37.5% |
| 1 | Romanticism | 35.7% |
| 9 | Realism | 35.2% |

## Summary for Draft

The WikiArt case study demonstrates four analytical contributions of the new pipeline:

1. **Cross-style formal affinities**: CLIP+HDBSCAN reveals visual similarities that
   cut across art-historical period labels. The "Baroque" cluster spans four centuries.
   Five distinct landscape traditions are discovered automatically.

2. **Bias as visible structure**: Ukiyo-e concentration (97.7% to one cluster) is
   even clearer with HDBSCAN than with KMeans. The density-based approach, which makes
   no assumptions about cluster count, still isolates non-Western art.

3. **Automatic semantic labeling**: CLIP-generated cluster labels ("Baroque," "Seascapes,"
   "Abstract Expressionist") provide immediate interpretive footholds that KMeans
   cluster numbers did not.

4. **Granular discovery**: HDBSCAN's 19 clusters reveal sub-categories (5 portrait types,
   5 landscape types) that KMeans' fixed 15 could not distinguish.

## Figures Needed (Regeneration Required)

All figures from the old draft showed UMAP projection with 15 KMeans clusters.
New figures must show t-SNE projection with 19 HDBSCAN clusters:

- **Figure 1**: t-SNE scatter view with 19 HDBSCAN clusters, colored by cluster,
  with CLIP-generated labels.
- **Figure 2**: Detail showing Ukiyo-e concentration at periphery.
- **Figure 3**: Cluster composition heatmap (19 clusters x 27 styles).
- **Figure 4**: REMOVED or replaced (was UMAP vs t-SNE; now both are t-SNE).
  Could instead show: cluster label overlay, or hotspot sidebar, or filter panel.
"""
    path = os.path.join(ARTICLE_DIR, "case-study", "findings.md")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Written findings.md: {len(content)} chars")


def write_todo():
    content = """# ImageSpace CHR Software Paper — Remaining Tasks

## Status
**Outline**: Complete but needs technical updates for new architecture.
**Case study**: Re-analyzed with new pipeline (19 HDBSCAN clusters, t-SNE).
**Draft**: REVISED for new architecture (2026-02-10). ~7,200 words — needs trimming to 6,000.
**Target**: "Computational Approaches to Art" special issue, *Computational Humanities Research* journal. Deadline **30 June 2026**.

---

## Completed Tasks
- [x] Process WikiArt with ImageSpace (49,585 images, CPU, 40.7 min)
- [x] Analyze cluster results (19 HDBSCAN clusters, findings.md updated)
- [x] Write full draft (draft.md — revised for new architecture)
- [x] First critique round (31 items addressed)
- [x] Second critique round (14 items addressed)
- [x] Technical reference updated for new architecture
- [x] Memory bank updated for new architecture
- [x] Update draft for new pipeline (UMAP->t-SNE, KMeans->HDBSCAN, 8->4 modes, etc.)

## Remaining Tasks

### 1. Trim Draft to Word Limit
- Current: ~7,200 words (body). Limit: 6,000 words.
- Need to cut ~1,200 words. Candidates:
  - Context section (§2): some theoretical discussion may compress
  - Related Software (§3): CLIP bias paragraph may partially merge with §5
  - Development Methodology (§4): pipeline stage list may abbreviate
  - Audience (§5): final paragraph may merge with §5.2

### 2. Generate New Figures
- [ ] Screenshot t-SNE scatter plot with 19 HDBSCAN cluster annotations (Figure 1)
- [ ] Screenshot showing Ukiyo-e cluster isolation in t-SNE (Figure 2)
- [ ] Cluster composition heatmap for 19 clusters (Figure 3)
- [ ] Figure 4: REPLACED (was UMAP vs t-SNE). Options:
  - Hotspot sidebar showing CLIP-generated labels
  - Filter panel demonstration
  - Detail panel with k-NN similar images
- [ ] Add figure captions and embeds to draft

### 3. Update Outline
- [ ] Update §4.1 (5 stages -> 9 stages)
- [ ] Update §4.3 (8 modes -> 4 modes)
- [ ] Update §4.5 case study data
- [ ] Update §5.2 constraints (10K -> 50K+)
- [ ] Update Abstract and PLS entries
- [ ] Update glossary (remove UMAP, add t-SNE, HDBSCAN, PCA, WebGL)
- [ ] Update Table 1 feature comparison

### 4. Add New References to Zotero
- [ ] Policar et al. 2019. openTSNE (bioRxiv 731877)
- [ ] Campello et al. 2013. HDBSCAN (PAKDD)
- [ ] van der Maaten & Hinton 2008. t-SNE (JMLR)
- [ ] (PixiJS — software reference, may not need Zotero entry)

### 5. Prepare Anonymous Repository
- [ ] Set up GitHub repo with anonymous URL
- [ ] Include ImageSpace source code, requirements, README
- [ ] Add case-study reproduction instructions

### 6. Mint DOI
- [ ] Zenodo deposit with CITATION.cff
- [ ] Add DOI to Data Availability Statement

### 7. Write Cover Letter
- [ ] Address CFP themes: "Computational Approaches to Art"
- [ ] Highlight: CPU-first accessibility, HDBSCAN automatic discovery, bias-as-finding

### 8. Final Submission Preparation
- [ ] Complete [To be completed] placeholders (Acknowledgements, Funding)
- [ ] Final anonymization pass
- [ ] Convert to submission format (LaTeX if required by CHR)
- [ ] Final word count verification (~6,000 max)

---

## Key Paths
| Resource | Path |
|---|---|
| Draft | `/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article/draft.md` |
| Outline | `/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article/outline.md` |
| Notes | `/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article/notes.md` |
| Technical Reference | `/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article/imagespace_technical_reference.md` |
| Memory system | `/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article/memory-system/` |
| ImageSpace code | `/Users/nabeel/Documents/ImageSpace/` |
| WikiArt images | `~/Documents/wikiart/` |
| Case study findings | `/Users/nabeel/Dropbox/Spring 2026/ImageSpace Article/case-study/findings.md` |
| Zotero collection | Key `HL77GI8M` |

## Key Constraints
- **Max 6,000 words / 12 pages**
- **Double-anonymous** — no author names, no "I"
- **Chicago Author-Date** citation style
- **Deadline: 30 June 2026**
- **Submission: ScholarOne** at mc.manuscriptcentral.com/ch-research
"""
    path = os.path.join(ARTICLE_DIR, "todo.md")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Written todo.md: {len(content)} chars")


if __name__ == "__main__":
    write_findings()
    write_todo()
    print("Done.")
