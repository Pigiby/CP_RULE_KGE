# CP_RULE_KGE

Rule learning on knowledge graphs with [PyClause](https://github.com/symbolic-kg/PyClause) (AMIE or AnyBURL), using the datasets of [LibKGE](https://github.com/uma-pi1/kge).

`kge/` and `PyClause/` are **not** part of this repository (they are in `.gitignore`). Clone and install them inside the repo root as described below.

## Requirements

- Python ≥ 3.10
- Java (for AMIE / AnyBURL): check with `java -version`, on macOS install with `brew install openjdk`
- A C++ compiler with OpenMP support (needed to build PyClause, see below)

## 1. Create a virtual environment

```bash
git clone <this-repo-url> CP_RULE_KGE
cd CP_RULE_KGE

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyyaml
```

If conda is active, run `conda deactivate` first so the venv is not built on top of it.

## 2. Install LibKGE

```bash
git clone https://github.com/uma-pi1/kge.git
cd kge
python -m pip install -e .
cd ..
```

## 3. Download the datasets

The datasets live in `kge/data/`. Each dataset is a folder with `train.txt`, `valid.txt` and `test.txt` (tab-separated `subject relation object`), plus files produced by LibKGE's preprocessing.

You can choose which datasets to download.

**Only one dataset** (e.g. WN18RR, the one used by default here):

```bash
cd kge/data
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wnrr.tar.gz
tar xvf wnrr.tar.gz
python preprocess/preprocess_default.py wnrr
cd ../..
```

Replace `wnrr` with any of the datasets below. Note that `fb15k` and `wn18` also need the `train.txt`/`valid.txt`/`test.txt` symlinks created in `download_all.sh`, so for those two it is easier to use the script.

**All datasets** (FB15k, FB15k-237, WN18, WN18RR, YAGO3-10):

```bash
cd kge/data
sh download_all.sh
cd ../..
```

To download only some of them with the script, delete or comment out the blocks of the datasets you don't want in `kge/data/download_all.sh`. Each dataset has its own block, starting with a comment such as `# wnrr`. Datasets that are already present are skipped.

| Folder | Dataset |
|---|---|
| `fb15k` | FB15k |
| `fb15k-237` | FB15k-237 |
| `wn18` | WN18 |
| `wnrr` | WN18RR |
| `yago3-10` | YAGO3-10 |

## 4. Install PyClause

```bash
git clone https://github.com/symbolic-kg/PyClause
python -m pip install ./PyClause
```

### macOS: `clang++: error: unsupported option '-fopenmp'`

Apple's `clang++` has no OpenMP support. Build with Homebrew's LLVM instead:

```bash
brew install llvm libomp

export CC="$(brew --prefix llvm)/bin/clang"
export CXX="$(brew --prefix llvm)/bin/clang++"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix libomp)/lib"

python -m pip install --no-cache-dir ./PyClause
```

If importing `c_clause` then fails with `Library not loaded: libomp.dylib`, also run `export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib"`.

### Check the installation

```bash
python -c "from clause import Learner, Options; from c_clause import Loader; print('ok')"
```

## 5. Learn rules

The settings are split over three files in `configs/rule_learners/`:

| File | Content |
|---|---|
| `rule_learner.yaml` | learner (`amie` or `anyburl`), output folder, confidence threshold, `swap_only` |
| `amie_config.yaml` | AMIE options, JVM options, loader options |
| `anyburl.yaml` | AnyBURL options and learning time, JVM options, loader options |

The datasets are chosen with `--dataset`, one or more folder names in `kge/data`. By default all five datasets are used. Rules are learned on `kge/data/<dataset>/train.txt`.

```bash
python src/rule_learner.py                             # all 5 datasets, settings from rule_learner.yaml
python src/rule_learner.py --dataset wnrr fb15k-237    # only some datasets
python src/rule_learner.py --learner anyburl           # override the learner
python src/rule_learner.py --min-confidence 0.8        # override the threshold
```

Rules are written to `rules/original_split/<dataset>/<learner>_rules.txt`. If that file already exists, it is reused unless `output.overwrite: true` is set in `rule_learner.yaml`.

From Python:

```python
from rule_learner import RuleLearner

rule_learner = RuleLearner("wnrr")          # dataset + configs/rule_learners/rule_learner.yaml
rules = rule_learner.learn()                # rules above the threshold
pairs = rule_learner.swap_pairs(rules)      # pairs R1, R2 with the head and one body atom swapped
```

### Rules file format

Each line has four tab-separated columns:

```
937    828    0.883671    _also_see(X,Y) <= _also_see(Y,X)
```

| Column | Meaning |
|---|---|
| predicted | number of body groundings in the training set (predictions made by the rule) |
| correct | predictions that are true in the training set (support) |
| confidence | correct / predicted |
| rule | `head <= body`; X and Y are the head variables, A, B, ... body-only variables |

If the confidence shows a decimal comma (`0,883671`), Java used a non-English locale. `read_rules` handles that. To get a decimal point, add `-Duser.language=en` and `-Duser.country=US` to `java_options` in the learner config.
