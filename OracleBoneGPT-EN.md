# Treating Oracle Bone Inscriptions as a Cipher

## What a Small Language Model Can Recover from Context and Rubbings

**Li Zhen (李臻)** · Independent researcher

Exploratory research report · 13 November 2026

### Abstract

Can a model recover the identity of a familiar oracle bone character class after its label is concealed? We use this question to distinguish sequence prediction, recovery of known correspondences, and evidence for a particular glyph on a rubbing. On three OBIMD splits grouped by catalogue identifiers and documented joins, a small GPT reduced next-character negative log likelihood relative to suffix retrieval (3.830 versus 3.984), while mean top-1 accuracy was 27.10% versus 27.37%. In 86 matched closed-list trials with 64 candidates, increasing the number of contexts from one to five raised GPT top-5 recovery from 49 to 66 trials. A follow-up on other bone groups reproduced the direction (53 to 68 of 80). In whole-class holdouts, GPT did not consistently beat a weighted bag of characters. On a later set of 90 queries it retrieved 40 correct references in the top five, against 30 for the bag baseline. Shuffling the two nearest ordinary characters on each side yielded an average of 40.47 hits. Visual evidence helped under artificial occlusion, while fixed-gallery tests and external rubbing examples revealed wrong-position predictions and confident guesses when the answer was absent. The experiments support evidence-linked candidate retrieval and leave the meanings of undeciphered signs unresolved.

**Keywords:** oracle bone inscriptions, computational decipherment, character-level language models, candidate retrieval, rubbings

### 1. Research question

A cipher suggests an experiment with a known answer. Replace a known character class with an anonymous identifier, retain the surrounding inscription, and ask whether examples from other bones help recover its correspondence. This design measures the value of observable context without assigning a meaning to an undeciphered sign. It assumes neither a one-to-one mapping to modern Chinese nor a known fraction of historical correspondences. Snyder, Barzilay and Knight used cognate evidence in work on Ugaritic [4]. The present experiments use within-corpus annotations and do not construct cross-period phonetic, graphic, or semantic correspondences.

We ask three questions. Does a small model predict annotated character sequences on held-out bones? Do multiple contexts improve ranking of concealed classes? Does adding visual evidence associate a candidate with the correct glyph box? These questions require different answers and denominators.

### 2. Materials and methods

We use the OBIMD release pinned to commit `4820d2f11a8f45f3347591fdcd5f6466706b0dce` [1]. Its JSON has 10,077 bone catalogue records, 21,941 annotated groups including 17,150 inscription groups, and 115,319 annotated positions including 21,667 explicit missing slots. The observed inventory includes 1,730 main classes and 2,747 subclasses. Catalogue records cannot be equated with distinct physical bones. Alphanumeric class labels, glyph subclasses, reference readings, uncertain marks, unlabelled boxes, and missing slots are retained separately. Dataset reference readings provide scoring labels and may carry uncertainty. The corpus licence is CC BY 4.0.

The initial seeds were 17, 29, and 43. We split by connected components of catalogue identifiers using 55 edges from 35 documented joins [2]. Exact ordinary-character strings, shared contiguous five-character strings, and selected one-edit variants were flagged. Unrecorded joins, alternate catalogue numbers and shorter formulas can survive screening. Later tests explicitly exclude components from earlier anonymous test splits as specified in each protocol.

The next-character GPT has two layers, four heads, width 64 and a maximum context of 80. Models were randomly initialized and trained for 2,000 steps with AdamW, validating every 250 steps. The three initial models had about 174,000–176,000 parameters and took about 5–6 seconds each including validation on an Apple M5 Pro using MLX/Metal and float32. Baseline smoothing and checkpoint selection used validation. Accuracy counts out-of-vocabulary targets as misses. Negative log likelihood is computed on targets in the training vocabulary. Bidirectional infilling is evaluated separately using held-out internal positions.

In anonymous recovery, forward and reverse models score the same candidate list. We first conceal identities at query time, leaving labelled training examples available. In a stricter task we replace every occurrence of the target classes in training and checkpoint validation by a shared placeholder. Both queries and reference contexts then use masked class identities. The GPT representation concatenates forward and reverse predictive distributions over the remaining vocabulary. Nearest-neighbour anchors and sided inverse-document-frequency weighted character counts are simpler alternatives. Square-root features are compared with cosine similarity and ties are scored conservatively in the reference-matching task.

For the visual task, glyph crops come from real rubbings. Artificial occlusion supplies a known target for the controlled trial. Fixed-gallery and external examples test transfer, wrong-position boxes, and missing candidates. Visual features include pixel, gradient and skeleton evidence. The fixed fusion adds half the standardized context score to the standardized visual score. External examples are tied to catalogue identifiers, published readings and image positions where available.

### 3. Results

#### 3.1 Sequence prediction and infilling

| Next-character method | Top 1 | Top 5 | Negative log likelihood |
|---|---:|---:|---:|
| Training frequency | 6.80% | 20.71% | 5.258 |
| Previous character | 22.59% | 46.59% | 4.081 |

| Previous two characters | 25.90% | 49.96% | 3.990 |
| Suffix retrieval | 27.37% | 50.29% | 3.984 |
| Small GPT | 27.10% | 49.32% | 3.830 |

GPT had lower loss in all three splits. Among records that did not trigger duplicate screening, GPT and suffix retrieval achieved 18.79% and 18.40% top-1 accuracy, with all three paired intervals including zero, and losses of 4.505 and 4.724. For classes observed 1–9 times during training, top-5 accuracy was 3.22% versus 5.28%. For short groups of 2–4 positions, top-1 was 18.62% versus 18.55%.

For infilling, the three splits contained 10,251, 9,448, and 11,305 masked positions. Mean top-1 accuracy was 28.69% for left retrieval, 35.16% for two-sided retrieval and 27.35% for the two-sided small model. The latter two reached 58.40% and 47.50% top-5. On records that did not trigger duplicate screening, their top-1 scores were 20.36% and 16.42%.

#### 3.2 Anonymous correspondence and the role of order

| Setting | Result | Qualification |
|---|---|---|
| 64 candidates, one versus five contexts | GPT top-5 rose from 49 to 66 of 86. With five contexts, trigram statistics reached 58 and reassigned contexts reached 10. | Five usable examples were required. The 86 trials share classes and bone components. Uniform random top-5 expectation is 7.81%. |
| New bones, one versus five contexts | GPT top-5 rose from 53 to 68 of 80. | The three lists share cases and form only four connected dependency groups. |
| Count the training-frequency prior once | Original sample: 66 to 78 top-5 hits. New sample: 30 to 45 top-1 and 68 to 69 top-5. | Corrected trigrams reached 43 top-1 and 66 top-5 in the new sample. The original top-5 improvement did not replicate. |
| Mid- and low-frequency classes, shared lists | GPT 48 top-1 and 88 top-5 versus trigram 43 and 81 of 126. | Top-5 difference 5.6 percentage points, descriptive interval −3.3 to +16.2. For 35 rare classes, the top-5 counts were 21 and 19. |
| Earlier whole-class holdout, three splits | GPT top-5 57/157, 60/144, 52/131 versus weighted bag 54/157, 58/144, 59/131. | All three paired intervals included zero. An ordered-training advantage over shuffled training was also inconsistent. |
| New whole-class holdout, 453 classes concealed | In 90 fixed 16-way trials, GPT reached 10 top-1 and 40 top-5; bag 9 and 30; adjacent anchors 4 and 18. | There were 266 nonshared query and reference bone components. GPT's top-5 difference against bag was +11.1 percentage points, interval 0 to +19.8. |
| Frozen model, query order perturbed | Original top-5 40/90. Local shuffling averaged 40.47, shuffling all ordinary characters within sides averaged 38.61. | In 64 repetitions, side, length, word counts and special positions stayed fixed. Reference contexts retained their original order. Both descriptive paired intervals included zero. |

Repeated contexts can concentrate scores even when paired with the wrong sign. On a separate 43-query check using later contexts to rerank proposed candidates, GPT reached 17 correct first choices and retaining the original first choice reached 18. These outcomes illustrate why a narrow score distribution alone does not validate a proposed correspondence.

#### 3.3 Candidate absence and abstention

With 63 candidates, a validation-set gate yielded 6 correct answers and 2 incorrect answers on 97 answer-present trials, and 7 answers on the 97 paired answer-absent trials. On the 126 mid- and low-frequency cases, GPT accepted 21 correct present cases and falsely answered 13 absent cases. Trigram statistics accepted 13 correct cases and falsely answered 6 absent cases.

In the new whole-class holdout, we held the list at 16 in both conditions by replacing the correct class with a fixed distractor. On 60 test cases, the maximum-similarity gate yielded 1 correct answer when the reference was present and 0 false answers on the paired absence cases. On a further 38 cases whose class was already absent from the fixed reference inventory, the same gate falsely answered 4. With the alternative margin gate, those counts were 1, 1 and 0, respectively, with only 3 answers on the 60 present trials. The 30 calibration cases and low coverage constrain interpretation.

#### 3.4 Visual evidence and position

Artificial occlusion on 1,070 real rubbing crops gave 367 correct top-1 answers for text alone. Frequency-adjusted fusion reached 472 at 50% visible box area and 516 at 75%. In 49 examples of the five/ten contrast at 75% visibility, text top-5 included the target 14 times. Expanding the pool to the union of text top-50 and visual top-20 raised inclusion to 44, while final frequency-adjusted top-1 was 13. Among three exact text confusions between these two signs, a forced binary visual choice corrected all three and open-pool reciprocal rank fusion corrected two. Box area is a proxy for occlusion and cannot be interpreted as a fraction of strokes retained.

| Reference domain | Method | Top 1/80 | Top 5/80 | Top-5 corrections / regressions | Neighbour box also lists target in top 5 |
|---|---|---:|---:|---:|---:|
| Rubbings | Visual | 4 | 12 | — | 2 |
| Rubbings | Statistical fusion | 5 | 15 | 6 / 3 | 6 |
| Rubbings | GPT fusion | 6 | 11 | 5 / 6 | 7 |
| Facsimiles | Visual | 10 | 19 | — | 0 |
| Facsimiles | Statistical fusion | 6 | 18 | 4 / 5 | 5 |
| Facsimiles | GPT fusion | 8 | 17 | 4 / 6 | 5 |

The 378-class galleries used two reference crops per class. Each fusion interval included zero. A separate 71-component stroke-alignment test increased visual top-20 counts by two and four depending on reference domain, with no net top-5 increase.

#### 3.5 External rubbing checks

Of four prespecified rare-class leads screened from a publication-linked set, two met inclusion criteria. The dragon sign remained outside the top 20. The sign transcribed as *ke* ranked sixth with the original average score against facsimiles and fifth with aligned skeletons. The other two leads were excluded because of a side mismatch and an unconfirmed catalogue reading. Earlier selected cases for *gao* and *ya* ranked third and second. Selected examples and the prespecified series have different denominators.

The date window in Heji 23520 occurred 45 times in training. Text ranked the expected sign first, while GPT fusion ranked it seventh in the correct rubbing box and first in a real neighbouring box. In Heji 8425, the reference class was absent from the gallery, and fusion confidently proposed a different class. Two impressions of the same bone, Heji 23589, produced different false suggestions for another absent class. These cases identify specific failure mechanisms and do not estimate a population error rate.

### 4. Discussion and conclusion

The cipher analogy led to a test with recoverable ground truth. More occurrences can improve candidate coverage on held-out bones. A substantial part of that information is available to simple retrieval or character co-occurrence statistics. The strongest evidence for GPT is improved next-character probability estimates and, in some settings, increased top-five coverage. The current tests leave its incremental ranking advantage uncertain and show little sensitivity to the query-order interventions studied here. Template families, incomplete join records, shared candidate groups, and the restriction to known labels all limit transfer.

The prototype links candidate glyphs and contexts to their sources for inspection. It has not established a useful automatic acceptance rule or measured reductions in research time. Genuine decipherment would require hypotheses with distinct, testable predictions and new evidence capable of distinguishing them. Independent impressions, chronological evidence, joins and historically justified character correspondences are possible sources. The present project cannot estimate the fraction of ancient signs corresponding to later characters.

### Data, code and AI assistance

The reproduction index identifies the archived splits, protocols, checksums and model entry points. Some local checkpoints and third-party images must be obtained or regenerated separately. Dataset licences are separate from manuscript rights. Third-party images are cited through source records. OpenAI Codex assisted with implementation, experiment execution and verification, drafting, translation, and typesetting. Numerical results were computed from data and model outputs by the scripts described in the reproduction index.

### References

1. Li, B., Yang, J., Liang, Y., et al. (2026). *OBIMD: A Multi-modal Dataset for Contextual Interpretation of Oracle Bone Inscriptions*. Scientific Data. [Article](https://doi.org/10.1038/s41597-026-06967-0) · [Pinned dataset](https://huggingface.co/datasets/KLOBIP/OBIMD/tree/4820d2f11a8f45f3347591fdcd5f6466706b0dce).
2. Zhang, H., Wang, T., Zhang, Z., et al. (2026). *A multi-modal dataset and method for bone-level association prediction in oracle bone inscriptions*. [Article](https://doi.org/10.1038/s40494-025-02282-w) · [Data](https://doi.org/10.5281/zenodo.17749727).
3. Li, G., Peng, S., Wan, X., et al. (2026). *Chronicles-OCR: A Cross-Temporal Perception Benchmark for the Evolutionary Trajectory of Chinese Characters*. [arXiv:2605.11960](https://arxiv.org/abs/2605.11960).
4. Snyder, B., Barzilay, R., and Knight, K. (2010). A Statistical Model for Lost Language Decipherment. *Proceedings of ACL*, 1048–1057. [ACL Anthology](https://aclanthology.org/P10-1107/).
5. *Jiaguwen Muben Daxi* (甲骨文摹本大系). [Electronic catalogue](https://eoracle.asia/).
6. Chen, P. (2020). Study of fine oracle bone rubbings comparing the Palace Museum Ma Heng volume and *Heji*. *Gugong Xuekan*, 21, figure 17. [Publication](https://www.dpm.org.cn/Uploads/File/2021/04/19/u607d2c4be8cd2.pdf).
