import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode  # noqa: E402

from baremetal_llm.scaling import SubwordTokenizer
from baremetal_llm.utils.paths import data_path, part_run_dir


def main():
    args = resolve_mode(part_parser("Part 4").parse_args())
    banner("Part 4: BPE tokenizer")

    vocab = 512 if args.demo else 2048
    tok = SubwordTokenizer(vocab=vocab)
    tok.train(data_path())

    out = part_run_dir(4, "part4-demo") / "tokenizer"
    tok.save(out)

    sample = "Transformers use self-attention to model sequences."
    ids = tok.encode(sample)
    print(f"tokens: {len(ids)}")
    print(f"round-trip: {tok.decode(ids)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
