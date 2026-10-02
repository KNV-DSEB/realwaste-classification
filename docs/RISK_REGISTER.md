# RISK_REGISTER

| Risk | Severity | Trigger | Mitigation | Owner | Status |
|---|---|---|---|---|---|
| Scope creep | Critical | New model/task proposed after Day 3 | Reject unless it fixes a material validity issue | Integration Lead | Open |
| Data leakage | Critical | Split recreated differently by model/member | One frozen split.csv reused everywhere | Data Lead | Open |
| Test-set tuning | Critical | Test metrics checked repeatedly during development | Validation-only model selection | Evaluation Lead | Open |
| Inconsistent preprocessing | High | Different model notebooks use different transforms | Shared transform/data module + spec | Integration Lead | Open |
| Colab disconnect | High | Runtime resets during training | Save best checkpoint/logs directly to Drive | Model Leads | Open |
| GPU unavailable | Medium | Colab GPU quota/unavailable | Kaggle as compute backup using same repo/config/split | Integration Lead | Open |
| Complex CNN overfits | Medium | train rises, val stalls/degrades | augmentation, BN/dropout, early stopping | Complex Lead | Open |
| Class imbalance harms minority classes | Medium | low minority recall / skewed counts | Macro-F1, per-class metrics; consider weighting after audit | Data/Eval | Open |
| Report starts too late | Critical | report <50% by Day 7 | Write sections in parallel from Day 1 | All | Open |
| Code freeze missed | Critical | architecture changes after Day 7 | Freeze; only bug/validity fixes allowed | Integration Lead | Open |
