"""Check user-facing CLI failures without invoking model inference."""
import builtins
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from solar_insight.cli import main


class CLITests(unittest.TestCase):
    def test_invalid_arguments_report_errors_without_outputs(self):
        cases = [
            ([], 1, "Cannot read image:"),
            (["--observed-at", "2025-04-30T14:08:10+09:00"], 1,
             "Solar axes need observed_at, latitude and longitude together"),
            (["--grid"], 1,
             "Heliographic grid needs observation time and observer coordinates"),
            (["--grid-spacing", "20"], 2, "invalid choice"),
            (["--checkpoint", "missing.pth"], 2,
             "Experimental inference needs --scaler and --allow-legacy-preprocessing"),
            (["--observed-at", "invalid", "--latitude", "34.965",
              "--longitude", "136.624"], 1,
             "Use an ISO observation timestamp and a valid timezone"),
        ]
        for options, status, message in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "outputs"
                result = subprocess.run(
                    [sys.executable, "-m", "solar_insight.cli",
                     str(Path(tmp) / "missing.png"), "--output-dir", str(output),
                     *options],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertFalse(output.exists())

    def test_missing_sunpy_reports_installation_command(self):
        real_import = builtins.__import__

        def without_sunpy(name, *args, **kwargs):
            if name == "sunpy" or name.startswith("sunpy."):
                raise ImportError("Simulated missing optional dependency")
            return real_import(name, *args, **kwargs)

        stderr = io.StringIO()
        argv = ["solar-insight", "missing.png",
                "--observed-at", "2025-04-30T14:08:10+09:00",
                "--latitude", "34.965", "--longitude", "136.624"]
        with patch.object(sys, "argv", argv), \
                patch("builtins.__import__", side_effect=without_sunpy), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Solar axes require:", stderr.getvalue())
        self.assertIn(".[solar]", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
