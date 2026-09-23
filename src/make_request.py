"""Create a compact JSON request for score.py."""
import argparse
import base64
import io
import json
from pathlib import Path
from PIL import Image


def main(image, out):
    photo = Image.open(image).convert("RGB")
    photo.thumbnail((512, 512))
    buffer = io.BytesIO()
    photo.save(buffer, "JPEG", quality=85, optimize=True)
    Path(out).write_text(json.dumps({"image": base64.b64encode(buffer.getvalue()).decode("ascii")}), encoding="utf-8")
    print(f"Wrote {out} ({len(buffer.getvalue())} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="sample-request.json")
    main(**vars(parser.parse_args()))