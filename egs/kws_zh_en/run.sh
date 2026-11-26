#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=../../:$PYTHONPATH

stage=0
stop_stage=100

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

exp_dir=zipformer/exp_tiny_cr_rnnt_all_data

# Model configuration of the released model
# sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 (~3M parameters).
model_args=(
  --decoder-dim 320
  --joiner-dim 320
  --num-encoder-layers 1,1,1,1,1,1
  --downsampling-factor 1,2,4,8,4,2
  --feedforward-dim 192,192,192,192,192,192
  --encoder-dim 128,128,128,128,128,128
  --encoder-unmasked-dim 128,128,128,128,128,128
  --cnn-module-kernel 15,15,15,15,15,15
)

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train the bilingual KWS model."
  if [ ! -e data/fbank/.wenetspeech.done ] || [ ! -e data/fbank/.gigaspeech.done ]; then
    log "You need to run prepare.sh first."
    exit 1
  fi

  python ./zipformer/train.py \
      --world-size 8 \
      --exp-dir $exp_dir \
      "${model_args[@]}" \
      --num-epochs 20 \
      --base-lr 0.04 \
      --lr-epochs 1.0 \
      --lr-batches 7500 \
      --use-fp16 1 \
      --start-epoch 1 \
      --seed 42 \
      --causal 1 \
      --chunk-size 8,16,32,-1 \
      --left-context-frames 64,128,256,-1 \
      --use-transducer 1 \
      --use-ctc 1 \
      --use-cr-ctc 1 \
      --ctc-loss-scale 0.1 \
      --cr-loss-scale 0.02 \
      --lm-scale 0.25 \
      --enable-spec-aug 0 \
      --time-mask-ratio 2.5 \
      --enable-musan 1 \
      --num-buckets 300 \
      --wenetspeech-subset L \
      --gigaspeech-subset XL \
      --mux-weights 0.5,0.5 \
      --tokens data/lang_phone/tokens.txt \
      --lexicon data/lang_phone/g2p.dict \
      --max-duration 1000
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Average checkpoints and export pretrained.pt"
  python ./zipformer/export.py \
      --exp-dir $exp_dir \
      --tokens data/lang_phone/tokens.txt \
      --epoch 13 \
      --avg 2 \
      --causal 1 \
      "${model_args[@]}"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Export streaming ONNX models (chunk-16/left-64, chunk-8/left-64, chunk-16/left-128)"
  for chunk in 16 8; do
    python ./zipformer/export-onnx-streaming.py \
        --exp-dir $exp_dir \
        --tokens data/lang_phone/tokens.txt \
        --epoch 13 \
        --avg 2 \
        --causal 1 \
        --chunk-size $chunk \
        --left-context-frames 64 \
        "${model_args[@]}"
  done

  python ./zipformer/export-onnx-streaming.py \
      --exp-dir $exp_dir \
      --tokens data/lang_phone/tokens.txt \
      --epoch 13 \
      --avg 2 \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      "${model_args[@]}"
fi
