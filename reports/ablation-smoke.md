# 消融表 · smoke

生成时间：2026-11-17 10:43:09

## 0. 运行条件

- label：smoke
- 题集：golden_set.jsonl
- 题数：3
- 语料目录：data\raw
- 向量库：kb__bge-m3-1024（90 chunk / 11 文档）
- Embedding：BAAI/bge-m3
- Reranker：BAAI/bge-reranker-v2-m3（enabled=True）
- 忠实度判定：rule

「答案 Hit@1」是在**回答正文的引用序列**上判的：期望文档必须真的出现在答案里，
光被召回不算。语料少于几十篇时两个口径会几乎相等，那说明的是题目太简单，不是效果好。

## 1. 总体

| 配置 | 候选 Hit@1 | 候选 Hit@3 | 答案 Hit@1 | 答案 MRR | 要点覆盖 | 误拒率 | 正确拒答率 | P50 |
|---|---|---|---|---|---|---|---|---|
| +rerank | 100.0% | 100.0% | 100.0% | 1.000 | 100.0% | 0.0% | 0.0% | 126ms |
| 关闸门 | 100.0% | 100.0% | 100.0% | 1.000 | 100.0% | 0.0% | 0.0% | 127ms |

## 2. 每行实际生效的配置

| 行 | 覆盖参数 | weight_vector | weight_bm25 | reranker | 词面覆盖 | 阈值 | claim_check |
|---|---|---|---|---|---|---|---|
| +rerank | （默认） | 1.0 | 1.0 | True | 0.3 | 0.55 | fix |
| 关闸门 | retrieval.min_lexical_coverage=0、generation.claim_check=off | 1.0 | 1.0 | True | 0.0 | 0.0 | off |
