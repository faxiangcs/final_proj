"""Regression tests for the legacy reward's diagnosis parser.

`medical_reward.py` imports ``swift.plugin``; the test stubs it so the parser can be
tested without ms-swift installed. This is a compatibility test for the old parser.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Stub ms-swift before importing medical_reward.
_swift = types.ModuleType("swift")
_plugin = types.ModuleType("swift.plugin")


class _ORM:
    pass


_plugin.ORM = _ORM
_plugin.orms = {}
sys.modules.setdefault("swift", _swift)
sys.modules.setdefault("swift.plugin", _plugin)

import medical_reward  # noqa: E402


class LegacyDiagnosisParserTest(unittest.TestCase):
    def test_final_with_colon_is_clean(self) -> None:
        text = (
            "【诊断推理】：考虑鉴别诊断包括 Metastasis; Abscess。"
            "结合特征，最终诊断为：Glioma。"
        )
        diff, final = medical_reward.extract_diagnosis(text)
        self.assertEqual(diff, "Metastasis; Abscess")
        self.assertEqual(final, "Glioma")

    def test_final_without_colon(self) -> None:
        text = (
            "【诊断推理】：考虑鉴别诊断包括 Metastasis。"
            "结合特征，最终诊断是 Glioma。"
        )
        _, final = medical_reward.extract_diagnosis(text)
        self.assertEqual(final, "Glioma")

    def test_final_with_ascii_colon(self) -> None:
        text = (
            "【诊断推理】：考虑鉴别诊断包括 Metastasis。"
            "结合特征，最终诊断: Glioma。"
        )
        _, final = medical_reward.extract_diagnosis(text)
        self.assertEqual(final, "Glioma")


if __name__ == "__main__":
    unittest.main()
