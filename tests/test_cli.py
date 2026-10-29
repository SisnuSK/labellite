from pathlib import Path

from typer.testing import CliRunner

from labellite import __version__
from labellite.cli import app

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
COMMANDS = (
    "extract",
    "probe",
    "finetune",
    "conformal",
    "shift",
    "activelearn",
    "explain",
    "review",
    "analyze",
    "headline",
)


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_review_command_tallies_a_sheet(tmp_path):
    sheet = tmp_path / "review_sample.csv"
    sheet.write_text("tile,id,true,pred,confidence,failure_type,notes\na,1,x,y,0.9,ambiguous,\n")
    result = runner.invoke(app, ["review", "--sheet", str(sheet)])
    assert result.exit_code == 0 and "| ambiguous | 1 | 100% |" in result.output


def test_invalid_config_fails(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: nope\ndataset: pcam\n")
    result = runner.invoke(app, ["extract", "--config", str(p)])
    assert result.exit_code not in (0, 2)
