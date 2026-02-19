from pathlib import Path
import re


def test_pyproject_license_is_mit():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert re.search(
        r'^\s*license\s*=\s*"MIT"\s*$|^\s*license\s*=\s*\{\s*text\s*=\s*"MIT"\s*\}\s*$',
        text,
        flags=re.MULTILINE,
    )


def test_license_file_exists_and_is_mit():
    license_file = Path(__file__).resolve().parents[1] / "LICENSE"
    text = license_file.read_text(encoding="utf-8")
    assert "MIT License" in text


def test_readme_license_section_not_tbd():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "## License" in text
    assert "TBD" not in text
