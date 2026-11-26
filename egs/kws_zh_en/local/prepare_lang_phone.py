#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corporation
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare data/lang_phone for bilingual (Chinese + English) KWS training.

The modeling units are the same as the released model
sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20:
  - Chinese: partial pinyin with tone (initial + final), e.g. 中国 -> zh ōng g uó
  - English: CMU phones with stress digits, e.g. HELLO -> HH AH0 L OW1

Both are produced by a single cppinyin Encoder built from g2p.dict:

    import cppinyin
    encoder = cppinyin.Encoder("data/lang_phone/g2p.dict")
    encoder.encode("我是中国人民的儿子 I LOVE CHINA", tone="normal", partial=True)

g2p.dict and tokens.txt are downloaded from the author's ModelScope repo
(the exact files used to train the released model). You can also pass
--g2p-dict / --tokens to use local copies.

Usage:
    pip install cppinyin==0.11
    ./local/prepare_lang_phone.py --lang-dir data/lang_phone
"""

import argparse
import logging
import urllib.request
from pathlib import Path

MODELSCOPE_REPO = "pkufool/icefall-kws-zipformer-zh-en-3M-2025-12-20"
MODELSCOPE_URL = (
    "https://www.modelscope.cn/api/v1/models/{repo}/repo"
    "?FilePath={path}&Revision=master"
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("data/lang_phone"),
        help="Output directory containing g2p.dict and tokens.txt",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=MODELSCOPE_REPO,
        help="ModelScope repo hosting data/lang_phone",
    )
    parser.add_argument(
        "--g2p-dict",
        type=Path,
        default=None,
        help="Optional local g2p.dict; skip downloading if given",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Optional local tokens.txt; skip downloading if given",
    )
    return parser


def download(repo: str, remote_path: str, dest: Path):
    if dest.is_file() and dest.stat().st_size > 0:
        logging.info(f"{dest} already exists, skipping download")
        return
    url = MODELSCOPE_URL.format(repo=repo, path=remote_path)
    logging.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".downloading")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main():
    args = get_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args.lang_dir.mkdir(parents=True, exist_ok=True)
    g2p_dict = args.lang_dir / "g2p.dict"
    tokens_txt = args.lang_dir / "tokens.txt"

    if args.g2p_dict is not None:
        g2p_dict.write_bytes(args.g2p_dict.read_bytes())
    else:
        download(args.repo, "data/lang_phone/g2p.dict", g2p_dict)

    if args.tokens is not None:
        tokens_txt.write_bytes(args.tokens.read_bytes())
    else:
        download(args.repo, "data/lang_phone/tokens.txt", tokens_txt)

    tokens = [line.split()[0] for line in tokens_txt.read_text().splitlines()]
    assert tokens[0] == "<blk>", tokens[:3]
    assert len(tokens) == len(set(tokens)), "duplicated tokens"
    logging.info(f"Vocabulary size: {len(tokens)} (including <blk>)")

    try:
        import cppinyin
    except ImportError:
        logging.warning(
            "cppinyin is not installed, skipping sanity check. "
            "Install it with: pip install cppinyin==0.11"
        )
        return

    encoder = cppinyin.Encoder(str(g2p_dict))
    token_set = set(tokens)
    for text in ("我是中国人民的儿子 I LOVE CHINA", "你好小明", "LIGHT UP"):
        encoded = encoder.encode(text, tone="normal", partial=True)
        oov = [t for t in encoded if t not in token_set]
        assert not oov, f"{text} produced OOV tokens: {oov}"
        logging.info(f"{text} -> {' '.join(encoded)}")

    logging.info(f"Done. lang dir: {args.lang_dir}")


if __name__ == "__main__":
    main()
