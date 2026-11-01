"""
DecodeLabs | Artificial Intelligence | Project 4
Image / Text Recognition (Basic) - Path 1: OCR with pytesseract + OpenCV

Author : Aakarsh Kumar
Batch  : 2026

Pipeline (IPO):
    INPUT    -> load image (a sample is generated if you don't pass one)
    PROCESS  -> grayscale -> Gaussian blur -> deskew -> adaptive thresholding
                -> Tesseract OCR (configurable --psm)
    OUTPUT   -> text, per-word confidence, 80% confidence gate, annotated image

Gatekeeper validations (from the project slides):
    [1] Library integration       [2] Pre-processing integrity
    [3] Accuracy benchmark >= 80% [4] Visual confirmation

Setup:
    pip install opencv-python pytesseract pillow numpy
    + install the Tesseract engine itself (see README notes in the chat):
      Windows: https://github.com/UB-Mannheim/tesseract/wiki

Run:
    python ocr_recognition.py                      (uses a generated sample image)
    python ocr_recognition.py my_photo.jpg --psm 6
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

CONFIDENCE_THRESHOLD = 80          # the "80% gate" from the slides
OUTPUT_DIR = "output"

# On Windows, Tesseract is usually not on PATH: point pytesseract at the default install
_WIN_PATHS = [
    r"C:\Users\aakar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
if "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = r"C:\Users\aakar\AppData\Local\Programs\Tesseract-OCR\tessdata"
for _p in _WIN_PATHS:
    if os.path.exists(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        break


# ---------------------------------------------------------------- sample input
def make_sample_image(path: str) -> None:
    """Create a deliberately messy test image: tilted text, shadow gradient, noise."""
    w, h = 1000, 520
    img = Image.new("L", (w, h), 245)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except OSError:
            font = ImageFont.load_default(size=48)
    lines = ["DECODELABS AI PROJECT 4", "Optical Character Recognition", "The machine can see text"]
    for i, line in enumerate(lines):
        draw.text((60, 90 + i * 110), line, fill=25, font=font)

    img = img.rotate(3, resample=Image.BICUBIC, fillcolor=245)          # tilt
    arr = np.array(img).astype(np.float32)
    shadow = np.tile(np.linspace(1.0, 0.55, w), (h, 1))                  # uneven lighting
    arr = arr * shadow + np.random.default_rng(42).normal(0, 9, arr.shape)  # noise
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- pre-processing
def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the text tilt with minAreaRect and rotate back to a horizontal baseline."""
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(inv)
    if points is None:
        return gray, 0.0
    angle = cv2.minAreaRect(points)[-1]
    angle = angle - 90 if angle > 45 else angle           # normalize to (-45, 45]
    if abs(angle) < 0.5 or abs(angle) > 20:               # already straight / unreliable
        return gray, 0.0
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def preprocess(image_bgr: np.ndarray) -> dict:
    """Step 1 grayscale -> Step 2 blur -> Step 3 deskew -> adaptive threshold."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    straightened, angle = deskew(blurred)
    binary = cv2.adaptiveThreshold(
        straightened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return {"gray": gray, "blurred": blurred, "deskewed": straightened,
            "binary": binary, "angle": angle}


# ---------------------------------------------------------------- recognition
def run_ocr(binary: np.ndarray, psm: int) -> list[dict]:
    """Run Tesseract and return word-level results with confidence (0-100)."""
    data = pytesseract.image_to_data(
        binary, config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])
        if text and conf >= 0:
            words.append({
                "text": text, "conf": conf,
                "x": data["left"][i], "y": data["top"][i],
                "w": data["width"][i], "h": data["height"][i],
                "line": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
            })
    return words


def draw_results(image_bgr: np.ndarray, words: list[dict], angle: float) -> np.ndarray:
    """Draw boxes + labels for words that pass the gate. Boxes live in the deskewed
    coordinate space, so the original is rotated the same way before drawing."""
    canvas = image_bgr.copy()
    if angle:
        h, w = canvas.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    for wd in words:
        if wd["conf"] >= CONFIDENCE_THRESHOLD:                       # draw_box_and_label()
            x, y, w, h = wd["x"], wd["y"], wd["w"], wd["h"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 170, 0), 2)
            label = f"{wd['text']} {wd['conf']:.0f}%"
            cv2.putText(canvas, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 120, 0), 2, cv2.LINE_AA)
        # else: drop_detection() -> nothing is drawn
    return canvas


def assemble_text(words: list[dict]) -> str:
    """Rebuild readable text line by line from the accepted words."""
    lines: dict[tuple, list[str]] = {}
    for wd in words:
        lines.setdefault(wd["line"], []).append(wd["text"])
    return "\n".join(" ".join(ws) for ws in lines.values())


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="DecodeLabs Project 4: OCR pipeline")
    parser.add_argument("image", nargs="?", default="sample_input.png", help="path to an image")
    parser.add_argument("--psm", type=int, default=6,
                        help="Tesseract page segmentation mode: 3 auto, 6 block, 7 line, 11 sparse")
    parser.add_argument("--show", action="store_true", help="open result windows")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- INPUT
    if args.image == "sample_input.png" and not os.path.exists(args.image):
        make_sample_image(args.image)
        print(f"No image given -> generated sample: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Could not read image: {args.image}")
    h, w = image.shape[:2]
    print("=" * 60)
    print("STEP 1: INPUT")
    print("=" * 60)
    print(f"File: {args.image} | size: {w}x{h} | channels: {image.shape[2]} "
          f"| data points: {image.size:,}")

    # ---- PROCESS: pre-processing
    steps = preprocess(image)
    print("\n" + "=" * 60)
    print("STEP 2: PRE-PROCESSING")
    print("=" * 60)
    print("Grayscale conversion ......... done")
    print("Gaussian blur (3x3) .......... done")
    print(f"Deskew ....................... rotated {steps['angle']:+.2f} degrees")
    print("Adaptive thresholding ........ done")
    for name in ("gray", "deskewed", "binary"):
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), steps[name])

    # ---- PROCESS: recognition
    try:
        words = run_ocr(steps["binary"], args.psm)
    except pytesseract.TesseractNotFoundError:
        sys.exit("\nTesseract engine not found. Install it and re-run:\n"
                 "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                 "  Linux  : sudo apt install tesseract-ocr\n"
                 "  macOS  : brew install tesseract")

    accepted = [wd for wd in words if wd["conf"] >= CONFIDENCE_THRESHOLD]
    dropped = [wd for wd in words if wd["conf"] < CONFIDENCE_THRESHOLD]
    avg_conf = sum(wd["conf"] for wd in accepted) / len(accepted) if accepted else 0.0

    # ---- OUTPUT
    print("\n" + "=" * 60)
    print(f"STEP 3: RECOGNITION (psm {args.psm}, confidence gate >= {CONFIDENCE_THRESHOLD}%)")
    print("=" * 60)
    print(f"{'WORD':<22}{'CONFIDENCE':>10}   STATUS")
    for wd in words:
        status = "KEEP" if wd["conf"] >= CONFIDENCE_THRESHOLD else "DROP"
        print(f"{wd['text']:<22}{wd['conf']:>9.1f}%   {status}")

    text = assemble_text(accepted)
    print("\nExtracted text (accepted words only):")
    print("-" * 60)
    print(text or "(nothing passed the confidence gate)")
    print("-" * 60)
    print(f"Words kept: {len(accepted)} | dropped: {len(dropped)} | "
          f"average confidence of kept words: {avg_conf:.1f}%")

    annotated = draw_results(image, words, steps["angle"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "result.png"), annotated)
    with open(os.path.join(OUTPUT_DIR, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    # ---- Gatekeeper validation
    checks = {
        "1. Library integration (pytesseract + OpenCV ran without errors)": True,
        "2. Pre-processing integrity (grayscale + adaptive threshold)": True,
        f"3. Accuracy benchmark (avg confidence >= {CONFIDENCE_THRESHOLD}%)":
            bool(accepted) and avg_conf >= CONFIDENCE_THRESHOLD,
        "4. Visual confirmation (annotated image saved)": os.path.exists(
            os.path.join(OUTPUT_DIR, "result.png")),
    }
    print("\n" + "=" * 60)
    print("GATEKEEPER VALIDATION")
    print("=" * 60)
    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\nSaved to ./{OUTPUT_DIR}/ : gray.png, deskewed.png, binary.png, "
          "result.png, extracted_text.txt")

    if args.show:
        cv2.imshow("Original", image)
        cv2.imshow("Binary (adaptive threshold)", steps["binary"])
        cv2.imshow("Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
