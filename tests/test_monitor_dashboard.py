"""Tests for scripts/monitor_dashboard.py helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.monitor_dashboard import _final_snippet, _read_jsonl


class MonitorDashboardTest(unittest.TestCase):
    def test_final_snippet_canonical(self):
        row = {
            "response": "【异常定位】：[1, 2, 3, 4]\n"
            "【诊断推理】：考虑鉴别诊断包括 A; B。结合特征，最终诊断为：Glioma。"
        }
        self.assertEqual(_final_snippet(row), "Glioma")

    def test_final_snippet_markdown(self):
        row = {
            "response": "### 3. 诊断推理\n**最终临床诊断**：\n"
            "**右顶叶脑膜瘤恶变（Anaplastic Meningioma）**。"
        }
        self.assertEqual(
            _final_snippet(row), "右顶叶脑膜瘤恶变（Anaplastic Meningioma）"
        )

    def test_final_snippet_falls_back_to_last_line(self):
        row = {"response": "line one\nlast line"}
        self.assertEqual(_final_snippet(row), "last line")

    def test_read_jsonl_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                '{"index": 0}\nnot json\n{"index": 1}\n', encoding="utf-8"
            )
            rows = _read_jsonl(path)
            self.assertEqual([row["index"] for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
