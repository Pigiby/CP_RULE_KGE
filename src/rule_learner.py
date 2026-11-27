from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any
import argparse
import re
import yaml
from clause import Learner, Options
from c_clause import Loader

from data_manager import DEFAULT_RATIOS, DataManager

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "rule_learners"
DEFAULT_CONFIG = CONFIG_DIR / "rule_learner.yaml"
DATA_DIR = ROOT / "kge" / "data"

SUPPORTED_LEARNERS = ("amie", "anyburl")
DATASETS = ("fb15k", "fb15k-237", "wn18", "wnrr", "yago3-10")
DEFAULT_LEARNER_CONFIGS = {
    "amie": CONFIG_DIR / "amie_config.yaml",
    "anyburl": CONFIG_DIR / "anyburl.yaml",
}
THRESHOLD_KEYS = ("min_confidence", "min_correct", "max_length", "variables_only")


_ATOM_RE = re.compile(r"([^\s(),]+)\(([^,()]+),([^,()]+)\)")
_VARIABLES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class Atom:
    """A single atom relation(subject, object); terms are variables (X, Y, A, ...) or entities."""

    relation: str
    subject: str
    object: str

    @property
    def has_constant(self) -> bool:
        return self.subject not in _VARIABLES or self.object not in _VARIABLES

    def __str__(self) -> str:
        return f"{self.relation}({self.subject},{self.object})"


@dataclass(frozen=True)
class Rule:
    """One line of an AnyBURL-format rules file: predicted, correct, confidence, head <= body."""

    predicted: int     # groundings of the body (number of predictions made on train)
    correct: int       # predictions that are true in train (support)
    confidence: float  # correct / predicted
    head: Atom
    body: tuple[Atom, ...]
    text: str          # the rule string exactly as in the file

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def has_constant(self) -> bool:
        return self.head.has_constant or any(a.has_constant for a in self.body)


def _parse_atoms(text: str) -> list[Atom]:
    return [Atom(r, s, o) for r, s, o in _ATOM_RE.findall(text)]


def passes_threshold(
    rule: Rule,
    min_confidence: float | None = None,
    min_correct: int | None = None,
    max_length: int | None = None,
    variables_only: bool = False,
) -> bool:
    """True if ``rule`` satisfies every filter that is not None / False."""
    if min_confidence is not None and rule.confidence < min_confidence:
        return False
    if min_correct is not None and rule.correct < min_correct:
        return False
    if max_length is not None and rule.length > max_length:
        return False
    if variables_only and rule.has_constant:
        return False
    return True


def read_rules(
    path_rules: str | Path,
    min_confidence: float | None = None,
    min_correct: int | None = None,
    max_length: int | None = None,
    variables_only: bool = False,
) -> list[Rule]:
    """Read an AnyBURL/AMIE (ofmt=anyburl) rules file into a list of ``Rule``.

    Handles decimal commas in the confidence column (``0,883671``), which Java
    writes under a non-English locale. Optional filters are applied while reading.
    """
    path_rules = Path(path_rules)
    rules: list[Rule] = []
    with path_rules.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path_rules}:{lineno}: expected 4 tab-separated columns, got {len(parts)}")
            predicted, correct, confidence, text = parts

            head_txt, sep, body_txt = text.partition(" <= ")
            head = _parse_atoms(head_txt)
            if not sep or len(head) != 1:
                raise ValueError(f"{path_rules}:{lineno}: cannot parse rule {text!r}")

            rule = Rule(
                predicted=int(predicted),
                correct=int(correct),
                confidence=float(confidence.replace(",", ".")),
                head=head[0],
                body=tuple(_parse_atoms(body_txt)),
                text=text,
            )
            if passes_threshold(rule, min_confidence, min_correct, max_length, variables_only):
                rules.append(rule)
    return rules


RuleKey = tuple[str, tuple[str, ...]]


def _is_variable(term: str) -> bool:
    return term in _VARIABLES


def _canonical_key(head: Atom, body: tuple[Atom, ...]) -> RuleKey:
    """A key that is equal for two rules iff they are the same up to renaming of the
    body-only variables and reordering of the body atoms. Head variables become X, Y."""
    head_map: dict[str, str] = {}
    for term, name in ((head.subject, "X"), (head.object, "Y")):
        if _is_variable(term) and term not in head_map:
            head_map[term] = name
    others = sorted({t for a in body for t in (a.subject, a.object) if _is_variable(t) and t not in head_map})

    def rename(atom: Atom, mapping: dict[str, str]) -> str:
        s = mapping.get(atom.subject, atom.subject)
        o = mapping.get(atom.object, atom.object)
        return f"{atom.relation}({s},{o})"

    best: tuple[str, ...] | None = None
    for perm in permutations(range(len(others))):
        mapping = {**head_map, **{v: f"V{i}" for v, i in zip(others, perm)}}
        candidate = tuple(sorted(rename(a, mapping) for a in body))
        if best is None or candidate < best:
            best = candidate
    return rename(head, head_map), best or ()


def rule_key(rule: Rule) -> RuleKey:
    return _canonical_key(rule.head, rule.body)


def swapped_keys(rule: Rule) -> list[tuple[int, RuleKey]]:
    """For every body atom b_i, the key of the rule b_i <= (body without b_i), head."""
    keys = []
    for i, atom in enumerate(rule.body):
        new_body = rule.body[:i] + rule.body[i + 1:] + (rule.head,)
        keys.append((i, _canonical_key(atom, new_body)))
    return keys


@dataclass
class SwapPair:
    """R1 = h <= b_1, ..., b_i, ..., b_n and its partner R2 = b_i <= b_1, ..., h, ..., b_n.

    ``head`` / ``body`` give R2 written with the *same variable names as R1*
    (h sits at ``position`` in the body), so every variable means the same thing in
    both rules. ``partner`` is R2 as it appears in the rules file (head renamed to X,Y),
    and ``var_map`` maps each variable of ``partner`` to the R1 variable it stands for.
    """

    rule: Rule
    partner: Rule
    position: int
    head: Atom
    body: tuple[Atom, ...]
    var_map: dict[str, str]

    @property
    def text(self) -> str:
        """R2 in R1's variable names."""
        return f"{self.head} <= " + ", ".join(str(a) for a in self.body)


def _rename_atom(atom: Atom, mapping: dict[str, str]) -> Atom:
    return Atom(atom.relation, mapping.get(atom.subject, atom.subject), mapping.get(atom.object, atom.object))


def _find_var_map(
    src_head: Atom, src_body: tuple[Atom, ...], dst_head: Atom, dst_body: tuple[Atom, ...]
) -> dict[str, str] | None:
    """A one-to-one renaming of the variables of (src_head, src_body) that turns it into
    (dst_head, dst_body), body compared as a multiset; None if there is none."""
    if src_head.relation != dst_head.relation or len(src_body) != len(dst_body):
        return None

    mapping: dict[str, str] = {}
    for s, d in ((src_head.subject, dst_head.subject), (src_head.object, dst_head.object)):
        if _is_variable(s) != _is_variable(d) or (not _is_variable(s) and s != d):
            return None
        if _is_variable(s):
            if mapping.get(s, d) != d or (s not in mapping and d in mapping.values()):
                return None
            mapping[s] = d

    def terms(atoms: tuple[Atom, ...]) -> set[str]:
        return {t for a in atoms for t in (a.subject, a.object) if _is_variable(t)}

    src_rest = sorted(terms(src_body) - set(mapping))
    dst_rest = sorted(terms(dst_body) - set(mapping.values()))
    if len(src_rest) != len(dst_rest):
        return None

    target = sorted(str(a) for a in dst_body)
    for perm in permutations(dst_rest):
        candidate = {**mapping, **dict(zip(src_rest, perm))}
        if sorted(str(_rename_atom(a, candidate)) for a in src_body) == target:
            return candidate
    return None


def find_swap_pairs(rules: list[Rule]) -> list[SwapPair]:
    """All pairs in ``rules`` where R2 is R1 with its head and body atom i swapped,
    with the variables kept consistent between the two rules.

    Each unordered pair is reported once. A rule that is its own swap (e.g. the
    symmetry rule r(X,Y) <= r(Y,X)) is not paired with itself.
    """
    index: dict[RuleKey, list[int]] = {}
    for j, rule in enumerate(rules):
        index.setdefault(rule_key(rule), []).append(j)

    pairs: list[SwapPair] = []
    seen: set[tuple[int, int]] = set()
    for i, rule in enumerate(rules):
        for pos, key in swapped_keys(rule):
            # R2 in R1's own variable names: b_i <= b_1, ..., h (at position i), ..., b_n
            head = rule.body[pos]
            body = rule.body[:pos] + (rule.head,) + rule.body[pos + 1:]
            for j in index.get(key, []):
                if j == i or (min(i, j), max(i, j)) in seen:
                    continue
                var_map = _find_var_map(rules[j].head, rules[j].body, head, body)
                if var_map is None:
                    continue
                seen.add((min(i, j), max(i, j)))
                pairs.append(SwapPair(rule, rules[j], pos, head, body, var_map))
    return pairs


def write_swap_pairs(pairs: list[SwapPair], path: str | Path) -> None:
    """Write swap pairs to a tab-separated file, one pair per line:
    position, R1 predicted/correct/confidence, R1, R2 predicted/correct/confidence,
    R2 in R1's variable names, R2 as in the rules file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# position\tr1_predicted\tr1_correct\tr1_confidence\tr1"
                "\tr2_predicted\tr2_correct\tr2_confidence\tr2\tr2_original\n")
        for p in pairs:
            r1, r2 = p.rule, p.partner
            f.write(f"{p.position}\t{r1.predicted}\t{r1.correct}\t{r1.confidence}\t{r1.text}"
                    f"\t{r2.predicted}\t{r2.correct}\t{r2.confidence}\t{p.text}\t{r2.text}\n")


def keep_swappable(rules: list[Rule]) -> list[Rule]:
    """Keep only the rules that have at least one swap partner in ``rules`` (order preserved)."""
    in_pair = {id(r) for p in find_swap_pairs(rules) for r in (p.rule, p.partner)}
    return [r for r in rules if id(r) in in_pair]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file that must contain a top-level mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in {path}, got {type(data).__name__}")
    return data


def _resolve(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


class RuleLearner:
    """Learns rules with AMIE or AnyBURL via PyClause and returns those above a threshold.

    Configuration is split over three YAML files:
      * the general config (``configs/rule_learners/rule_learner.yaml``): which learner,
        output location and threshold (the dataset is passed to the constructor);
      * one learner-specific config per learner (``amie_config.yaml`` / ``anyburl.yaml``),
        referenced from the general config under ``learner_configs``.

    Recognised keys in the learner-specific config (all optional):
        learner: <name>                  sanity check, must match ``learner_name``
        java_options: [...]              -> learner.<name>.java_options
        <name>.raw / learning.raw        -> learner.<name>.raw.*
        <name>.time / learning.time      -> learner.anyburl.time (AnyBURL only)
        loader / inference.loader        -> loader.*
        learner.* / loader.* (flat keys) -> set as-is
    Keys whose value is null are skipped, so PyClause keeps its default.
    """

    def __init__(
        self,
        dataset: str,
        config_path: str | Path = DEFAULT_CONFIG,
        learner_name: str | None = None,
        path_train: str | Path | None = None,
        split_name: str = "original_split",
    ) -> None:
        """
        :param dataset: name of a dataset folder in ``kge/data`` (e.g. 'wnrr', 'fb15k-237').
        :param path_train: triples to learn the rules on; default ``kge/data/<dataset>/train.txt``.
            Usually ``DataManager(...).path("rule")``.
        :param split_name: subfolder of the rules dir, e.g. 'original_split' or 'transductive_split';
            rules go to ``<rules_dir>/<split_name>/<dataset>/<learner>_rules.txt``.
        :param config_path: the general rule learner config.
        :param learner_name: 'amie' or 'anyburl'; overrides ``learner`` in the general config.
        """
        self.config_path = Path(config_path)
        self.config: dict[str, Any] = _load_yaml(self.config_path)

        learner_name = (learner_name or self.config.get("learner") or "").lower()
        if learner_name not in SUPPORTED_LEARNERS:
            raise ValueError(
                f"learner must be one of {SUPPORTED_LEARNERS}, got {learner_name!r} "
                f"(set 'learner' in {self.config_path} or pass learner_name)"
            )
        self.learner_name = learner_name

        learner_configs = self.config.get("learner_configs") or {}
        if learner_name in learner_configs:
            self.learner_config_path = _resolve(learner_configs[learner_name], self.config_path.parent)
        else:
            self.learner_config_path = DEFAULT_LEARNER_CONFIGS[learner_name]

        self.dataset_name: str = dataset
        self.path_train: Path = Path(path_train) if path_train is not None else DATA_DIR / dataset / "train.txt"
        self.split_name = split_name

        output = self.config.get("output") or {}
        rules_dir = _resolve(output.get("rules_dir", "rules"), ROOT)
        self.path_rules: Path = rules_dir / split_name / self.dataset_name / f"{self.learner_name}_rules.txt"
        self.path_pairs: Path = self.path_rules.with_name(f"{self.learner_name}_swap_pairs.txt")
        self.overwrite: bool = bool(output.get("overwrite", False))

        threshold = self.config.get("threshold") or {}
        unknown = set(threshold) - set(THRESHOLD_KEYS)
        if unknown:
            raise ValueError(f"Unknown threshold keys in {self.config_path}: {sorted(unknown)}")
        self.threshold: dict[str, Any] = {k: threshold.get(k) for k in THRESHOLD_KEYS}
        self.threshold["variables_only"] = bool(self.threshold["variables_only"])
        self.swap_only: bool = bool(self.config.get("swap_only", False))

        self.loader: Loader | None = None
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        """(Re)load the learner-specific config and rebuild the PyClause Options and Learner."""
        self.learner_config: dict[str, Any] = _load_yaml(self.learner_config_path)

        cfg_learner = self.learner_config.get("learner")
        if isinstance(cfg_learner, str) and cfg_learner.lower() != self.learner_name:
            raise ValueError(
                f"Config {self.learner_config_path} is for learner {cfg_learner!r}, "
                f"but RuleLearner was created with {self.learner_name!r}"
            )

        self.options = Options()
        self.options.set("learner.mode", self.learner_name)
        self._apply_config(self.learner_config)
        self.learner = Learner(options=self.options.get("learner"))

    def _set(self, key: str, value: Any) -> None:
        if value is not None:
            self.options.set(key, value)

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        """Apply learner-specific YAML keys into PyClause Options for ``self.learner_name``."""
        name = self.learner_name
        prefix = f"learner.{name}"

        learner_section = cfg.get(name) if isinstance(cfg.get(name), dict) else {}
        learning_section = cfg.get("learning") if isinstance(cfg.get("learning"), dict) else {}

        # Raw options passed straight to the learner (AMIE CLI flags / AnyBURL properties)
        for section in (learning_section, learner_section):
            raw = section.get("raw")
            if isinstance(raw, dict):
                for key, value in raw.items():
                    self._set(f"{prefix}.raw.{key}", value)

        # AnyBURL learning time in seconds
        if name == "anyburl":
            for section in (learning_section, learner_section):
                self._set(f"{prefix}.time", section.get("time"))

        # JVM options
        java_options = cfg.get("java_options", learner_section.get("java_options"))
        if isinstance(java_options, list):
            self._set(f"{prefix}.java_options", java_options)

        # Loader options (top-level "loader" or "inference.loader")
        inference = cfg.get("inference") if isinstance(cfg.get("inference"), dict) else {}
        for loader_cfg in (inference.get("loader"), cfg.get("loader")):
            if isinstance(loader_cfg, dict):
                for key, value in loader_cfg.items():
                    opt_key = key if key.startswith("loader.") else f"loader.{key}"
                    self._set(opt_key, value)

        # Flat dot-path overrides at top level
        for key, value in cfg.items():
            if key.startswith(("learner.", "loader.")):
                self._set(key, value)

    def reload_config(self) -> None:
        """Reload the learner-specific config from disk and rebuild the Learner."""
        self._build()

    # ------------------------------------------------------------------ main entry point
    def learn(self, swap_only: bool | None = None, **threshold_overrides: Any) -> list[Rule]:
        """Mine rules (unless a rules file exists and overwrite is false) and return
        the rules that pass the threshold.

        ``threshold_overrides`` (min_confidence, min_correct, max_length, variables_only)
        replace the values from the general config for this call. If ``swap_only``
        (default: ``swap_only`` in the config), only rules with a swap partner among
        the thresholded rules are returned (see ``find_swap_pairs``).
        """
        if not self.path_train.exists():
            available = sorted(p.parent.name for p in DATA_DIR.glob("*/train.txt"))
            raise FileNotFoundError(
                f"Training triples not found: {self.path_train}\n"
                f"Datasets available in {DATA_DIR}: {available or 'none'} (see README, 'Download the datasets')"
            )

        if self.overwrite or not self.path_rules.exists():
            print(f"{self}: mining rules on {self.path_train} ...")
            self.mine_rules(self.path_train, self.path_rules)
            print(f"Rules written to {self.path_rules}")
        else:
            print(f"Reusing existing rules in {self.path_rules} (set output.overwrite: true to mine again)")

        rules = self.read_rules(self.path_rules, **threshold_overrides)
        if self.swap_only if swap_only is None else swap_only:
            rules = keep_swappable(rules)
        return rules

    @staticmethod
    def swap_pairs(rules: list[Rule]) -> list[SwapPair]:
        """Pairs R1, R2 where R2 is R1 with its head and body atom i swapped (same variables)."""
        return find_swap_pairs(rules)

    # ------------------------------------------------------------------ building blocks
    def mine_rules(
        self,
        path_train: str | Path,
        path_rules_output: str | Path,
    ) -> Path:
        """Run the rule learner on training triples and write rules to disk."""
        output_path = Path(path_rules_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.learner.learn_rules(
            path_data=str(path_train),
            path_output=str(output_path),
        )
        return output_path

    def read_rules(self, path_rules: str | Path, **threshold_overrides: Any) -> list[Rule]:
        """Parse a rules file and keep the rules that pass the threshold
        (general config values, replaced by any ``threshold_overrides``)."""
        unknown = set(threshold_overrides) - set(THRESHOLD_KEYS)
        if unknown:
            raise TypeError(f"Unknown threshold arguments: {sorted(unknown)}")
        return read_rules(path_rules, **{**self.threshold, **threshold_overrides})

    def load_rules(
        self,
        path_train: str | Path,
        path_rules: str | Path,
        reload: bool = True,
    ) -> Loader:
        """Load KG triples and mined rules into a c_clause.Loader."""
        if self.loader is None or reload:
            self.loader = Loader(options=self.options.get("loader"))
            self.loader.load_data(data=str(path_train))

        self.loader.load_rules(rules=str(path_rules))
        return self.loader

    def __repr__(self) -> str:
        return (
            f"RuleLearner(learner_name={self.learner_name!r}, "
            f"learner_config={str(self.learner_config_path)!r})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn rules with AMIE or AnyBURL and keep those above a threshold.")
    parser.add_argument("--dataset", nargs="+", default=list(DATASETS),
                        help=f"one or more dataset folders in kge/data (default: all of {list(DATASETS)}); "
                             "rules are learned on the original split and on the transductive split")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="general rule learner config")
    parser.add_argument("--learner", choices=SUPPORTED_LEARNERS, default=None, help="override 'learner' in the config")
    parser.add_argument("--min-confidence", type=float, default=None, help="override threshold.min_confidence")
    args = parser.parse_args()

    overrides = {"min_confidence": args.min_confidence} if args.min_confidence is not None else {}
    summary: list[tuple[str, str, int, int]] = []

    for dataset in args.dataset:
        # original split: rules on the original train; transductive split: rules on the 0.3 'rule' split
        for ratios in (None, DEFAULT_RATIOS):
            data = DataManager(dataset, ratios=ratios)
            print(f"\n{'=' * 20} {data} {'=' * 20}")
            rule_learner = RuleLearner(
                dataset=dataset,
                config_path=args.config,
                learner_name=args.learner,
                path_train=data.path("rule"),
                split_name=data.split_name,
            )
            rules = rule_learner.learn(**overrides)
            pairs = rule_learner.swap_pairs(rules)
            write_swap_pairs(pairs, rule_learner.path_pairs)

            print(f"{len(rules)} rules pass the threshold {rule_learner.threshold | overrides}"
                  f"{' and have a swap partner' if rule_learner.swap_only else ''}")
            print(f"{len(pairs)} swap pairs written to {rule_learner.path_pairs}")
            summary.append((dataset, data.split_name, len(rules), len(pairs)))

    print(f"\n{'dataset':<12}{'split':<22}{'rules':>8}{'pairs':>8}")
    for dataset, split_name, n_rules, n_pairs in summary:
        print(f"{dataset:<12}{split_name:<22}{n_rules:>8}{n_pairs:>8}")

if __name__ == "__main__":
    main()
