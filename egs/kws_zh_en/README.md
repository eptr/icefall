# kws_zh_en

Reproduction of the training recipe for the bilingual (Chinese + English)
open-vocabulary keyword spotting model
[sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html).

## Provenance

The released model was trained by the author (pkufool / Wei Kang) with a
private fork of icefall; the training code was never published as a public
recipe (see [icefall#2077](https://github.com/k2-fsa/icefall/issues/2077)).
This recipe reconstructs it from:

- the public KWS recipes `egs/wenetspeech/KWS` and `egs/gigaspeech/KWS`
  ([PR #1428](https://github.com/k2-fsa/icefall/pull/1428)), which provide the
  tiny streaming zipformer transducer code (the `model.py` there already
  contains the CR-CTC implementation);
- the CR-CTC training wiring from `egs/librispeech/ASR/zipformer/train.py`;
- the metadata embedded in the released PyTorch checkpoint
  ([ModelScope](https://www.modelscope.cn/models/pkufool/icefall-kws-zipformer-zh-en-3M-2025-12-20)),
  which records the exact hyper-parameters:
  `exp_dir=zipformer/exp_tiny_cr_rnnt_all_data`, `use_cr_ctc=True`,
  `ctc_loss_scale=0.1`, `cr_loss_scale=0.02`, `lm_scale=0.25`,
  `chunk_size=8,16,32,-1`, `left_context_frames=64,128,256,-1`,
  `num_epochs=20`, `base_lr=0.04`, `lr_epochs=1.0`, `max_duration=1000`,
  `num_buckets=300`, `time_mask_ratio=2.5`, `enable_spec_aug=False`,
  `tokens=data/lang_phone/tokens.txt`, `lexicon=data/lang_phone/g2p.model`.

## Modeling units

`data/lang_phone` (263 tokens) combines:

- Chinese: partial pinyin with tone, e.g. `中国` -> `zh ōng g uó`
- English: CMU phones with stress digits, e.g. `HELLO` -> `HH AH0 L OW1`

Both are produced by one [cppinyin](https://github.com/pkufool/cppinyin)
encoder built from `g2p.dict`:

```python
import cppinyin
encoder = cppinyin.Encoder("data/lang_phone/g2p.dict")
encoder.encode("我是中国人民的儿子 I LOVE CHINA", tone="normal", partial=True)
```

`g2p.dict` and `tokens.txt` are downloaded from the author's ModelScope repo
(the exact files used for the released model). English text must be upper
case; punctuation is stripped before encoding.

## Data

Training uses a 50/50 mux of:

- WenetSpeech L (Chinese, 10000 h), prepared via `egs/wenetspeech/ASR`
- GigaSpeech XL (English, 10000 h), prepared via `egs/gigaspeech/ASR`

Validation is WenetSpeech DEV + GigaSpeech DEV. MUSAN noise augmentation is
enabled. The original checkpoint does not record the exact mux weights;
`0.5,0.5` matches the equal corpus sizes and can be changed with
`--mux-weights`.

## Usage

```bash
cd egs/kws_zh_en

# 1. Prepare data (downloads WenetSpeech + GigaSpeech, computes fbank,
#    downloads data/lang_phone). Requires lots of disk and time.
./prepare.sh

# 2. Train (8 GPUs, ~3M params, 20 epochs)
./run.sh --stage 0 --stop-stage 0

# 3. Average epoch 12-13 and export pretrained.pt + streaming ONNX models
./run.sh --stage 1 --stop-stage 2
```

Requirements: `pip install cppinyin==0.11` in addition to the usual icefall
dependencies (k2, lhotse, torch, ...).

The exported ONNX models work with the sherpa-onnx keyword spotter. To
customize keywords, generate the keywords file with
`sherpa-onnx-cli text2token --tokens-type phone+ppinyin --lexicon data/lang_phone/g2p.dict`.

## Files

- `prepare.sh`: data preparation (WenetSpeech, GigaSpeech, lang_phone)
- `local/prepare_lang_phone.py`: download/verify `data/lang_phone`
- `zipformer/train.py`: training with cppinyin tokenization + CR-CTC
- `zipformer/asr_datamodule.py`: bilingual mux datamodule
- `zipformer/model.py`, `zipformer/zipformer.py`, ...: unchanged from
  `egs/wenetspeech/KWS/zipformer`
- `zipformer/export.py`, `zipformer/export-onnx-streaming.py`: checkpoint
  averaging and ONNX export
- `run.sh`: train / export stages

## Notes

- `decode.py` / `finetune.py` (KWS evaluation and keyword finetuning) are not
  included yet; they can be adapted from `egs/wenetspeech/KWS/zipformer` by
  replacing `text_to_pinyin` with the cppinyin encoder the same way
  `train.py` does.
- The released model was trained with `world_size=8`; reduce `--world-size`
  and `--max-duration` for smaller setups.
