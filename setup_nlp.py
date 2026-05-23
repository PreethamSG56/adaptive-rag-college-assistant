"""
setup_nlp.py
============
One-shot setup script for the NLP + Handwritten Notes pipeline.

Run this ONCE after `pip install -r requirements.txt`:
    python setup_nlp.py

What it does:
  1. Verifies all required packages are installed
  2. Downloads the spaCy English model (en_core_web_sm)
  3. Pre-warms the EasyOCR model cache (downloads weights once)
  4. Tests the full NLP pipeline on a short sample string
  5. Prints a summary of what is available
"""

import sys
import subprocess


def check_package(name: str) -> bool:
    try:
        __import__(name.replace("-", "_"))
        return True
    except ImportError:
        return False


def install_package(name: str) -> bool:
    print(f"  Installing {name}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", name, "-q"],
        capture_output=True,
    )
    return result.returncode == 0


def download_spacy_model() -> bool:
    print("  Downloading spaCy en_core_web_sm model...")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=False,
    )
    return result.returncode == 0


def prewarm_easyocr():
    """Download EasyOCR model weights once so users don't wait on first query."""
    print("  Pre-warming EasyOCR (downloading model weights if needed)...")
    try:
        import easyocr
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = easyocr.Reader(['en'], gpu=False, verbose=True)
        print("  ✅ EasyOCR ready.")
        return True
    except Exception as e:
        print(f"  ⚠️  EasyOCR pre-warm failed: {e}")
        return False


def test_nlp_pipeline():
    """Run the full NLP pipeline on sample OCR text."""
    print("\n  Testing NLP pipeline...")
    sample_ocr_text = (
        "Th1s 1s a handwrltten note about operat1ng systems. "
        "Process management 1ncludes schedul1ng CPU resources. "
        "Memory management uses virtual addressing and pag1ng."
    )
    try:
        sys.path.insert(0, ".")
        from src.nlp_processor import process_ocr_text
        result = process_ocr_text(
            sample_ocr_text,
            run_spell_correction=True,
            run_spacy=True,
        )
        print(f"  Original: {sample_ocr_text[:80]}...")
        print(f"  Cleaned:  {result.cleaned_text[:80]}...")
        print(f"  OCR confidence: {result.ocr_confidence:.2f}")
        print(f"  Keywords: {result.keywords[:5]}")
        print(f"  Spell corrected: {result.spell_corrected}")
        print(f"  spaCy enriched: {result.spacy_enriched}")
        return True
    except Exception as e:
        print(f"  ⚠️  NLP pipeline test failed: {e}")
        return False


def test_hallucination_guard():
    """Test the hallucination guard with a sample."""
    print("\n  Testing Hallucination Guard...")
    try:
        sys.path.insert(0, ".")
        from src.hallucination_guard import guard_answer
        from src.ingestion import Document

        # Mock context document
        doc = Document(
            text="Operating systems manage CPU and memory resources through scheduling algorithms.",
            source="test.pdf",
            chunk_id=0,
            metadata={"ocr_confidence": 0.85}
        )
        context = [(doc, 0.92)]

        # Test with grounded answer
        grounded_answer = "CPU scheduling is managed by the operating system."
        result = guard_answer(grounded_answer, context)
        print(f"  Grounded answer verdict: {result.verdict} (confidence: {result.overall_confidence:.2f})")

        # Test with hallucinated answer
        hallucinated = "The quantum computer uses blockchain to manage processes in Mars."
        result2 = guard_answer(hallucinated, context)
        print(f"  Hallucinated answer verdict: {result2.verdict} (confidence: {result2.overall_confidence:.2f})")
        return True
    except Exception as e:
        print(f"  ⚠️  Guard test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  College Notes RAG — NLP Setup")
    print("=" * 60)

    packages = {
        "easyocr": "easyocr",
        "symspellpy": "symspellpy",
        "spacy": "spacy",
        "cv2": "opencv-python-headless",
        "PIL": "Pillow",
    }

    print("\n[1] Checking packages...")
    all_ok = True
    for import_name, pip_name in packages.items():
        if check_package(import_name):
            print(f"  ✅ {pip_name} — installed")
        else:
            print(f"  ❌ {pip_name} — NOT found, installing...")
            if install_package(pip_name):
                print(f"  ✅ {pip_name} — installed successfully")
            else:
                print(f"  ❌ {pip_name} — installation FAILED")
                all_ok = False

    print("\n[2] Downloading spaCy language model...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("  ✅ en_core_web_sm already downloaded")
        except OSError:
            if download_spacy_model():
                print("  ✅ en_core_web_sm downloaded")
            else:
                print("  ⚠️  spaCy model download failed — spell segmentation will be disabled")
    except ImportError:
        print("  ⚠️  spaCy not installed — skipping model download")

    print("\n[3] Pre-warming EasyOCR...")
    prewarm_easyocr()

    print("\n[4] Running NLP pipeline test...")
    test_nlp_pipeline()

    print("\n[5] Running Hallucination Guard test...")
    test_hallucination_guard()

    print("\n" + "=" * 60)
    print("  Setup complete! You can now run:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
