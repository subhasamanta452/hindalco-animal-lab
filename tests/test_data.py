from pathlib import Path
from PIL import Image


ROOT = Path(__file__).parents[1] / "data" / "animals"


def test_animal_data_is_well_formed():
    assert ROOT.is_dir()
    folders = sorted(path for path in ROOT.iterdir() if path.is_dir())
    assert len(folders) >= 2
    assert all(folder.name == folder.name.lower() and " " not in folder.name for folder in folders)
    assert not any(path.is_file() for path in ROOT.iterdir())
    for folder in folders:
        images = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        assert len(images) >= 10
        for path in images:
            with Image.open(path) as image:
                image.verify()