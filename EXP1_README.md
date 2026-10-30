# EXP1: group preferences

Run commands from `TIGER-exp` using the existing TIGER Python environment.
Use the exact content embedding tensor that produced the selected SID mapping.
The example below uses Beauty; replace the dataset and paths together.

```bash
python build_group_features.py \
  --embeddings 'data/Beauty/text_embed_sentence-t5-base.pt' \
  --indices 'tokenizer/sid_result/Beauty_sentence-t5-base/K=256_D=3_index_nc.json' \
  --output 'data/Beauty/group_features.pt' \
  --group_prefix_len 2

bash train.sh --dataset Beauty --token_type sid_nc \
  --use_group_preferences --group_features 'data/Beauty/group_features.pt' \
  --group_prefix_len 2

bash evaluation.sh --dataset Beauty --token_type sid_nc \
  --use_group_preferences --group_features 'data/Beauty/group_features.pt' \
  --group_prefix_len 2 --results_file results/exp1_beauty.json
```

Omit `--use_group_preferences` and `--group_features` for the baseline.
EXP1 checkpoints live in the `exp1` subdirectory of the normal checkpoint
location. Use `--output_dir` (training) and `--ckpt_dir` (evaluation) to isolate
additional runs. Evaluation `--sid_prefix_len` independently controls metric
grouping; its default remains all semantic SID layers.

Fixed features contain item groups, group means, residuals and the source SID
mapping. Dataset loading verifies the mapping and grouping prefix. Rebuild the
features whenever embeddings or SID assignments change. Content embeddings and
residuals are fixed; only the projection and T5 parameters are trained.

Each sample first selects and truncates its raw item history. The same history
produces both SID tokens and per-group mean residuals. The collator pads the
residuals, masks padding and preserves ascending group-ID order in all modes. T5
projects the residuals and appends them to SID embeddings. Generation encodes
this combined input once and reuses it for constrained beam search. Targets
remain complete item SIDs, including the conflict-resolution token.

`max_sent_len` limits SID tokenization; EXP1 raises an error rather than silently
truncating a history already used to compute preferences. The combined encoder
length additionally includes at most `max_len` preference vectors.

```bash
python -m unittest discover -s tests -v
```
