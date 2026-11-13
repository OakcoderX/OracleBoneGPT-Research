# Oracle Bone GPT · 甲骨文 GPT

Exploratory experiments on sequence prediction, anonymous character-class recovery, and evidence retrieval from oracle bone rubbings. The study tests known correspondences and documents failure cases, including confident predictions when the correct class is absent.

- English: [Report PDF](output/pdf/OracleBoneGPT-EN.pdf) · [Editable manuscript](OracleBoneGPT-EN.md)
- 中文：[结案报告 PDF](output/pdf/OracleBoneGPT-ZH.pdf) · [可编辑正文](甲骨文GPT论文初稿.md)
- Evidence and methods / 数据与方法：[Reproduction index](论文复现索引.md) · [Data provenance and attribution](来源与许可.md)

## Reproduce the reported counts

Use Python 3.12 and NumPy 2.5.3, then run `python paper/verify_cipher.py` and `python paper/verify_closeout.py` from the repository root. These commands read the archived trial records and do not require a GPU. Model training used MLX 0.32.2 and Apple Metal. The source index provides the pinned OBIMD and join data references. This public snapshot includes selected scripts and evidence, and the index also points to experiments that require additional local checkpoints, visual utilities, or original image archives.

## What the study found

More occurrences improved recovery of known correspondences. The small GPT assigned better next-character probabilities than suffix retrieval, and its ranking advantage over statistical methods was inconsistent. A local prototype linked suggestions to examples that can be inspected. The experiments provide no independently verified readings of undeciphered signs.

研究以可揭晓答案的匿名字类做盲测。多条辞例有助于找回正确候选，小 GPT 的排序优势仍不稳定。原拓错框、候选缺席和弃答结果随报告公开。
