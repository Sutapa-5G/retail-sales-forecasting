"""Repository hygiene: no machine-specific paths, no placeholders, no random splitting, no training in the app."""
import json
import re

from src import config

ROOT = config.PROJECT_ROOT
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "raw"}
THIS_FILE = "test_project_hygiene.py"


def _text_files(suffixes):
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes and not (set(path.parts) & SKIP_DIRS) and path.name != THIS_FILE:
            yield path


def _find(pattern, suffixes):
    regex = re.compile(pattern)
    return [str(p.relative_to(ROOT)) for p in _text_files(suffixes) if regex.search(p.read_text(errors="ignore"))]


def test_no_machine_specific_absolute_paths():
    bad = "|".join(["/ho" + "me/", "/Us" + "ers/", "C:" + r"\\\\Us", "/m" + "nt/", "/tm" + "p/"])
    assert _find(bad, {".py", ".md", ".ipynb", ".txt", ".json", ".csv"}) == []


def test_no_placeholder_markers():
    markers = "|".join(["TO" + "DO", "FIX" + "ME", "YOUR_" + "PATH_HERE", "INSERT_" + "PATH", "PLACE" + "HOLDER"])
    assert _find(markers, {".py", ".md", ".ipynb", ".txt"}) == []


def test_no_random_splitting_anywhere_in_the_source():
    source = [p for p in _text_files({".py"}) if p.parts[-2] in {"src", "app"}]
    banned = re.compile("|".join(["train_test_" + "split", "K" + "Fold", "shuffle" + r"\s*=\s*True", r"\.sam" + "ple" + r"\(frac"]))
    assert [str(p.relative_to(ROOT)) for p in source if banned.search(p.read_text())] == []


def test_the_dashboard_never_trains_or_fits_anything():
    import ast
    text = (ROOT / "app" / "app.py").read_text()
    assert ".fit(" not in text and "fit_transform" not in text
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert "src.train" not in imported
    assert not any(name.startswith("sklearn") or name.startswith("xgboost") for name in imported)


def test_all_notebooks_are_valid_json_with_executed_outputs_and_no_errors():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [n.name for n in notebooks] == ["01_data_exploration.ipynb", "02_time_series_analysis.ipynb",
                                           "03_feature_engineering.ipynb", "04_model_training_evaluation.ipynb",
                                           "05_forecast_analysis.ipynb"]
    for path in notebooks:
        nb = json.loads(path.read_text())
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code_cells, path.name
        assert all(c["execution_count"] is not None for c in code_cells), f"{path.name} has unexecuted cells"
        for cell in code_cells:
            assert not any(o.get("output_type") == "error" for o in cell["outputs"]), f"{path.name} has an error output"


def test_gitignore_excludes_raw_data_environments_and_caches():
    text = (ROOT / ".gitignore").read_text()
    for required in ("data/raw/*", ".venv/", "__pycache__/", ".pytest_cache/", ".ipynb_checkpoints/", ".env"):
        assert required in text


def test_requirements_are_pinned():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)
