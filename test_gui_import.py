"""Test that GUI module can be imported."""

try:
    from src.gui import FrizzAnalysisGUI, main
    print("[OK] GUI module imports successfully")
    print("[OK] FrizzAnalysisGUI class found")
    print("[OK] main function found")
    print("\nReady to launch GUI!")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()

