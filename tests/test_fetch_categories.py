"""
The Amazon corpus must draw evenly from every requested category.

The fetcher used to fill the whole quota from whichever category it read
first, so a 20k-review corpus was 100% All_Beauty while its manifest listed
five categories - and the generated paper would have repeated that claim.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "data" / "fetch_real_datasets.py"


@pytest.fixture
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_real_datasets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_stream(category, n=3000):
    for i in range(n):
        yield {"rating": (1, 3, 5)[i % 3],
               "title": f"{category} title {i}",
               "text": f"a long enough review body number {i} from {category}"}


def test_every_category_contributes_equally(fetch, monkeypatch):
    categories = ["A", "B", "C"]
    monkeypatch.setattr(fetch, "resolve_amazon_files",
                        lambda: [(c, f"url-{c}") for c in categories])

    import datasets
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda *a, data_files=None, **k: fake_stream(data_files[-1]))

    result = fetch.build_amazon(total_rows=900, smoke=False)
    stats = result["stats"]

    assert stats["categories"] == categories
    all_rows = [r for split in result["splits"].values() for r in split]
    per_category = {c: sum(r["category"] == c for r in all_rows) for c in categories}
    assert set(per_category.values()) == {300}, per_category


def test_manifest_lists_only_categories_actually_used(fetch, monkeypatch):
    monkeypatch.setattr(fetch, "resolve_amazon_files",
                        lambda: [("A", "url-A"), ("EMPTY", "url-EMPTY")])

    import datasets

    def stream(*a, data_files=None, **k):
        return fake_stream("A") if data_files.endswith("A") else iter(())

    monkeypatch.setattr(datasets, "load_dataset", stream)
    stats = fetch.build_amazon(total_rows=300, smoke=False)["stats"]
    assert stats["categories"] == ["A"]
    assert stats["categories_requested"] == ["A", "EMPTY"]
