from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_path(name: str = "tiny_corpus.txt") -> Path:
    bundled = Path(__file__).resolve().parents[1] / "data" / name
    if bundled.exists():
        return bundled
    return repo_root() / "data" / name


def part_run_dir(part: int, name: str) -> Path:
    out = repo_root() / "parts" / f"part_{part}" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    return out
