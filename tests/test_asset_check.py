from pathlib import Path

from tools.check_assets import asset_rows, summarize


def test_asset_summary_detects_missing_required_asset(tmp_path: Path):
    present = tmp_path / "present.bin"
    present.write_bytes(b"ok")
    missing = tmp_path / "missing.bin"
    config = {
        "models": {
            "ready_model": {
                "assets": [
                    {
                        "kind": "url",
                        "path": str(present),
                        "required_for_smoke": True,
                    }
                ]
            },
            "missing_model": {
                "assets": [
                    {
                        "kind": "url",
                        "path": str(missing),
                        "required_for_smoke": True,
                    }
                ]
            },
        }
    }
    summary = summarize(asset_rows(config, ["ready_model", "missing_model"]))
    assert summary["required_assets"] == 2
    assert summary["ready_required_assets"] == 1
    assert summary["missing_required_assets"] == 1
    assert summary["models_with_missing_assets"] == ["missing_model"]
