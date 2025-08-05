from typer.testing import CliRunner
from lego_classifier.cli import app

runner = CliRunner()

def test_generate_happy_path():
    result = runner.invoke(app, ["run-inference", ".\data\inference\3022 Plate 2x2"])
    print(result.output)
    assert result.exit_code == 0