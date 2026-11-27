from __future__ import annotations
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "kge" / "data"
SPLITS_DIR = ROOT / "splits"

SPLITS = ("train", "rule", "valid", "calib", "test")
DEFAULT_RATIOS = {"train": 0.4, "rule": 0.3, "calib": 0.1, "test": 0.1, "valid": 0.1}

Triple = tuple[str, str, str]


def read_triples(path: Path) -> list[Triple]:
    """Read a tab-separated 'subject relation object' file."""
    with path.open(encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


def write_triples(triples: list[Triple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines("\t".join(t) + "\n" for t in triples)


class DataManager:
    """Builds the splits train / rule / valid / calib / test of a dataset in kge/data.

    ratios=None (original split):
        train = rule = original train  (KGE training and rule learning both on train)
        valid = original valid         (hyperparameter tuning)
        calib, test = original test split in half
    ratios given (transductive split):
        train + valid + test are merged, shuffled and split with the ratios, e.g.
        DEFAULT_RATIOS = train 0.4, rule 0.3, calib 0.1, test 0.1, valid 0.1.
        Triples whose entities or relation never appear in train are moved to train,
        so every other split only contains entities and relations seen in train.
    """

    def __init__(self, dataset: str, ratios: dict[str, float] | None = None, seed: int = 0) -> None:
        self.dataset = dataset
        self.ratios = ratios
        self.seed = seed
        self.split_name = "original_split" if ratios is None else "transductive_split"

        dataset_dir = DATA_DIR / dataset
        original = {name: read_triples(dataset_dir / f"{name}.txt") for name in ("train", "valid", "test")}
        rng = random.Random(seed)

        if ratios is None:
            test = original["test"][:]
            rng.shuffle(test)
            half = len(test) // 2
            self.splits = {
                "train": original["train"],
                "rule": original["train"],
                "valid": original["valid"],
                "calib": test[:half],
                "test": test[half:],
            }
        else:
            self.splits = self._ratio_split(original["train"] + original["valid"] + original["test"], ratios, rng)

    @staticmethod
    def _ratio_split(triples: list[Triple], ratios: dict[str, float], rng: random.Random) -> dict[str, list[Triple]]:
        if set(ratios) != set(SPLITS):
            raise ValueError(f"ratios must have exactly the keys {SPLITS}, got {sorted(ratios)}")
        if abs(sum(ratios.values()) - 1.0) > 1e-9:
            raise ValueError(f"ratios must sum to 1, got {sum(ratios.values())}")

        triples = triples[:]
        rng.shuffle(triples)
        splits: dict[str, list[Triple]] = {}
        start = 0
        for i, name in enumerate(SPLITS):
            end = len(triples) if i == len(SPLITS) - 1 else start + round(ratios[name] * len(triples))
            splits[name] = triples[start:end]
            start = end

        # transductive: entities and relations of every split must be seen in train
        entities = {e for s, _, o in splits["train"] for e in (s, o)}
        relations = {r for _, r, _ in splits["train"]}
        for name in SPLITS[1:]:
            kept = []
            for s, r, o in splits[name]:
                if s in entities and o in entities and r in relations:
                    kept.append((s, r, o))
                else:
                    splits["train"].append((s, r, o))
                    entities.update((s, o))
                    relations.add(r)
            splits[name] = kept
        return splits

    @property
    def split_dir(self) -> Path:
        return SPLITS_DIR / self.split_name / self.dataset

    def path(self, split: str) -> Path:
        """Write a split to splits/<split_name>/<dataset>/<split>.txt and return its path."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        path = self.split_dir / f"{split}.txt"
        write_triples(self.splits[split], path)
        return path

    def save(self) -> None:
        """Write all splits to splits/<split_name>/<dataset>/."""
        for split in SPLITS:
            write_triples(self.splits[split], self.split_dir / f"{split}.txt")

    def __getitem__(self, split: str) -> list[Triple]:
        return self.splits[split]

    def __repr__(self) -> str:
        sizes = ", ".join(f"{s}={len(self.splits[s])}" for s in SPLITS)
        return f"DataManager({self.dataset!r}, {self.split_name}: {sizes})"
