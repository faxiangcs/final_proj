"""Unit tests for the canonical Nova schema.

Run:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.nova_schema import (  # noqa: E402
    NovaTarget,
    clean_differential_list,
    clean_final_diagnosis,
    parse_label,
    parse_prediction,
    render_box_label,
    render_box_prompt,
    render_label,
    render_prompt,
    validate_target,
)


class FinalDiagnosisCleanTest(unittest.TestCase):
    def test_strips_boilerplate_prefixes(self) -> None:
        cases = {
            "Regarding the whole file, we believe it is falx EMH": "falx EMH",
            "This patient was diagnosed as having isolated agenesis of the septum pellucidum":
                "isolated agenesis of the septum pellucidum",
            "Laboratory tests and MRI confirmed toxoplasmosis of the optic nerve":
                "toxoplasmosis of the optic nerve",
            "Surgically and histologically confirmed haemorrhagic colloid cyst":
                "haemorrhagic colloid cyst",
            "Case of Wilson disease in a patient with giddiness": "Wilson disease",
            "Final diagnosis: Meningioma": "Meningioma",
        }
        for raw, expected in cases.items():
            self.assertEqual(clean_final_diagnosis(raw), expected)

    def test_leaves_real_names_untouched(self) -> None:
        for name in (
            "Pilocytic astrocytoma",
            "Congenital TORCH infection (most likely CMV)",
            "Multiple System Atrophy - Cerebellar type",
            # The disease is after "with"; trimming would delete the answer.
            "Adult onset seizures in a patient with neurocysticercosis",
        ):
            self.assertEqual(clean_final_diagnosis(name), name)


class IntegerBoxTest(unittest.TestCase):
    def test_integer_rendering_rounds_and_round_trips(self) -> None:
        target = NovaTarget(
            boxes=[[10.4, 20.6, 100.49, 120.51]],
            caption="Lesion.",
            final_diagnosis="Glioma",
            differential=["Metastasis"],
        )
        label = render_label(target, integer_boxes=True)
        self.assertIn("【异常定位】：[10, 21, 100, 121]", label)
        parsed = parse_label(label)
        self.assertEqual(parsed.boxes, [[10.0, 21.0, 100.0, 121.0]])
        self.assertNotIn(".00", label)

    def test_float_rendering_stays_default(self) -> None:
        target = NovaTarget(
            boxes=[[10.5, 20.5, 100.5, 120.5]],
            caption="Lesion.",
            final_diagnosis="Glioma",
            differential=["Metastasis"],
        )
        self.assertIn("[10.50, 20.50, 100.50, 120.50]", render_label(target))

    def test_box_only_prompt_and_label(self) -> None:
        prompt = render_box_prompt("A 50-year-old with headache.")
        self.assertIn("只输出异常定位", prompt)
        self.assertIn("【A 50-year-old with headache.】", prompt)
        self.assertEqual(
            render_box_label([[1.2, 2.8, 3.4, 4.6]], integer_boxes=True),
            "【异常定位】：[1, 3, 3, 5]",
        )


class SchemaRoundTripTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        target = NovaTarget(
            boxes=[[10.0, 20.0, 100.0, 120.0], [200.0, 210.0, 300.0, 320.0]],
            caption="Axial FLAIR shows a hyperintense lesion in the left frontal lobe.",
            final_diagnosis="Glioma",
            differential=["Metastasis", "Abscess"],
        )
        label = render_label(target)
        parsed = parse_label(label)

        self.assertEqual(parsed.caption, target.caption)
        self.assertEqual(parsed.final_diagnosis, target.final_diagnosis)
        self.assertEqual(parsed.differential, target.differential)
        self.assertEqual(len(parsed.boxes), 2)
        for got, want in zip(parsed.boxes, target.boxes):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w)
        self.assertFalse(parsed.legacy_format)
        self.assertEqual(validate_target(parsed), {})

    def test_prompt_contains_three_sections(self) -> None:
        prompt = render_prompt("A 50-year-old with headache.")
        self.assertIn("【A 50-year-old with headache.】", prompt)
        self.assertIn("1. 异常定位", prompt)
        self.assertIn("2. 图像描述", prompt)
        self.assertIn("3. 诊断推理", prompt)

    def test_prompt_fixes_language_and_coordinate_scale(self) -> None:
        prompt = render_prompt("history")
        self.assertIn("一律使用英文", prompt)
        self.assertIn("0–480 像素", prompt)
        self.assertIn("不要编号", prompt)

    def test_numbered_differentials_are_cleaned(self) -> None:
        text = (
            "【异常定位】：[10, 20, 100, 120]\n"
            "【图像描述】：lesion\n"
            "【诊断推理】：考虑鉴别诊断包括：1. Glioma；2. Metastasis；"
            "3. Abscess。结合特征，最终诊断为：Meningioma。"
        )
        parsed = parse_prediction(text)
        self.assertEqual(parsed.differential, ["Glioma", "Metastasis", "Abscess"])

    def test_lettered_differentials_are_cleaned(self) -> None:
        text = (
            "【异常定位】：[10, 20, 100, 120]\n"
            "【图像描述】：lesion\n"
            "【诊断推理】：考虑鉴别诊断包括 Cortically based neoplasms:; "
            "a.Dysembryonic neuroepithelial tumour (DNET); b.Ganglioglioma。"
            "结合特征，最终诊断为：Diffuse glioma。"
        )
        parsed = parse_prediction(text)
        self.assertEqual(
            parsed.differential,
            ["Dysembryonic neuroepithelial tumour (DNET)", "Ganglioglioma"],
        )
        self.assertEqual(parsed.final_diagnosis, "Diffuse glioma")

    def test_english_headings_and_final_are_parsed(self) -> None:
        text = (
            "1. Localization: [10, 20, 100, 120]\n\n"
            "2. Image Description: Axial T2 shows a left frontal lesion.\n\n"
            "3. Diagnostic Reasoning: Lymphoma; Metastasis. Given the history, "
            "the final diagnosis is: Meningioma."
        )
        parsed = parse_prediction(text)
        self.assertEqual(parsed.boxes, [[10.0, 20.0, 100.0, 120.0]])
        self.assertEqual(parsed.caption, "Axial T2 shows a left frontal lesion.")
        self.assertEqual(parsed.final_diagnosis, "Meningioma")
        self.assertEqual(parsed.differential, ["Lymphoma", "Metastasis"])

    def test_final_trailing_period_is_normalized(self) -> None:
        target = NovaTarget(
            boxes=[[10.0, 20.0, 100.0, 120.0]],
            caption="lesion",
            final_diagnosis="Glioma.",
            differential=["Metastasis"],
        )
        parsed = parse_label(render_label(target))
        self.assertEqual(parsed.final_diagnosis, "Glioma")

    def test_clean_differential_drops_final_with_period(self) -> None:
        diffs = clean_differential_list("Glioma.; Metastasis; Abscess", "Glioma.")
        self.assertEqual(diffs, ["Metastasis", "Abscess"])


class ParserTest(unittest.TestCase):
    def test_numbered_prediction(self) -> None:
        text = (
            "1. 异常定位：[10, 20, 100, 120]\n"
            "2. 图像描述：Axial FLAIR shows a lesion.\n"
            "3. 诊断推理：考虑鉴别诊断包括 Metastasis; Abscess。"
            "结合特征，最终诊断为：Glioma。"
        )
        target = parse_prediction(text)
        self.assertEqual(target.boxes, [[10.0, 20.0, 100.0, 120.0]])
        self.assertEqual(target.caption, "Axial FLAIR shows a lesion.")
        self.assertEqual(target.final_diagnosis, "Glioma")
        self.assertEqual(target.differential, ["Metastasis", "Abscess"])
        self.assertFalse(target.legacy_format)

    def test_legacy_five_diagnosis_format(self) -> None:
        text = (
            "【异常定位】：[10, 20, 100, 120]\n"
            "【图像描述】：Axial FLAIR shows a lesion.\n"
            "【诊断推理】：Glioma; Metastasis; Abscess"
        )
        target = parse_prediction(text)
        self.assertEqual(target.final_diagnosis, "Glioma")
        self.assertEqual(target.differential, ["Metastasis", "Abscess"])
        self.assertTrue(target.legacy_format)

    def test_json_prediction(self) -> None:
        text = (
            '{"boxes": [[10, 20, 100, 120]], "caption": "lesion", '
            '"final_diagnosis": "Glioma", "differential": ["Metastasis"]}'
        )
        target = parse_prediction(text)
        self.assertEqual(target.boxes, [[10.0, 20.0, 100.0, 120.0]])
        self.assertEqual(target.final_diagnosis, "Glioma")
        self.assertEqual(target.differential, ["Metastasis"])

    def test_missing_section_strict(self) -> None:
        with self.assertRaises(ValueError):
            parse_label("【异常定位】：[10, 20, 100, 120]\n【诊断推理】：Glioma", strict=True)

    def test_missing_section_lenient(self) -> None:
        target = parse_label("no sections at all", strict=False)
        self.assertEqual(target.boxes, [])
        self.assertEqual(target.caption, "")
        self.assertTrue(target.legacy_format)


class ValidationTest(unittest.TestCase):
    def _base(self, **overrides) -> NovaTarget:
        values = dict(
            boxes=[[10.0, 20.0, 100.0, 120.0]],
            caption="lesion",
            final_diagnosis="Glioma",
            differential=["Metastasis"],
        )
        values.update(overrides)
        return NovaTarget(**values)

    def test_valid(self) -> None:
        self.assertEqual(validate_target(self._base()), {})

    def test_not_found_and_empty(self) -> None:
        errors = validate_target(
            self._base(final_diagnosis="Final Diagnosis not found.", differential=[])
        )
        self.assertIn("final", errors)
        self.assertIn("differential", errors)

    def test_out_of_range_box(self) -> None:
        errors = validate_target(self._base(boxes=[[0.0, 0.0, 500.0, 100.0]]))
        self.assertIn("boxes", errors)

    def test_invalid_extent(self) -> None:
        errors = validate_target(self._base(boxes=[[100.0, 100.0, 100.0, 120.0]]))
        self.assertIn("boxes", errors)

    def test_too_many_boxes(self) -> None:
        errors = validate_target(
            self._base(
                boxes=[
                    [10.0, 20.0, 100.0, 120.0],
                    [200.0, 20.0, 300.0, 120.0],
                    [10.0, 200.0, 100.0, 300.0],
                    [200.0, 200.0, 300.0, 300.0],
                ]
            )
        )
        self.assertIn("boxes", errors)

    def test_duplicate_box(self) -> None:
        errors = validate_target(
            self._base(
                boxes=[[10.0, 20.0, 100.0, 120.0], [10.0, 20.0, 100.0, 120.0]]
            )
        )
        self.assertIn("boxes", errors)

    def test_duplicate_differential(self) -> None:
        errors = validate_target(
            self._base(differential=["Metastasis", "Metastasis"])
        )
        self.assertIn("differential", errors)

    def test_differential_equals_final(self) -> None:
        errors = validate_target(self._base(differential=["Glioma"]))
        self.assertIn("differential", errors)

    def test_too_many_differential(self) -> None:
        errors = validate_target(
            self._base(differential=["A", "B", "C", "D", "E"])
        )
        self.assertIn("differential", errors)

    def test_empty_caption(self) -> None:
        errors = validate_target(self._base(caption=""))
        self.assertIn("caption", errors)


if __name__ == "__main__":
    unittest.main()
