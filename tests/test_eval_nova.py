"""Tests for src/eval_nova.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.eval_nova import (
    convert_boxes,
    corpus_bleu,
    detect_coord_mode,
    diagnosis_keyword_match,
    diagnosis_match,
    evaluate_rows,
    greedy_match,
    load_diagnosis_aliases,
    localization_metrics,
    normalize_diagnosis,
    read_rows,
    scale_boxes,
)
from src.nova_schema import render_label, NovaTarget


def _label(boxes, caption="Axial T2 hyperintense lesion.", final="Meningioma",
           diffs=("Lymphoma", "Metastasis")):
    return render_label(
        NovaTarget(boxes=boxes, caption=caption, final_diagnosis=final,
                   differential=list(diffs))
    )


class CoordTest(unittest.TestCase):
    def test_detect_modes(self):
        self.assertEqual(detect_coord_mode([[[0.1, 0.2, 0.3, 0.4]]]), "norm1")
        self.assertEqual(detect_coord_mode([[[10, 20, 30, 40]]]), "norm100")
        self.assertEqual(detect_coord_mode([[[100, 200, 300, 400]]]), "pixel")
        self.assertEqual(detect_coord_mode([[[100, 200, 900, 800]]]), "norm1000")
        self.assertEqual(detect_coord_mode([[]]), "pixel")

    def test_convert_norm1000(self):
        boxes, stats = convert_boxes([[125, 250, 500, 480]], "norm1000")
        self.assertAlmostEqual(boxes[0][0], 60.0, places=3)
        self.assertAlmostEqual(boxes[0][3], 230.4, places=3)
        self.assertEqual(stats["dropped"], 0)

    def test_convert_swaps_and_clips(self):
        boxes, stats = convert_boxes([[300, 200, 100, 50]], "pixel")
        self.assertEqual(boxes[0], [100.0, 50.0, 300.0, 200.0])
        self.assertEqual(stats["swapped"], 2)

        boxes, stats = convert_boxes([[-10, -10, 600, 600]], "pixel")
        self.assertEqual(boxes[0], [0.0, 0.0, 480.0, 480.0])
        self.assertEqual(stats["clipped"], 1)

    def test_convert_drops_degenerate(self):
        boxes, stats = convert_boxes([[10, 10, 10.5, 300]], "pixel")
        self.assertEqual(boxes, [])
        self.assertEqual(stats["dropped"], 1)


class BoxScaleTest(unittest.TestCase):
    def test_identity(self):
        boxes = [[100, 100, 200, 200]]
        self.assertEqual(scale_boxes(boxes, 1.0), boxes)

    def test_scales_about_center(self):
        boxes = scale_boxes([[100, 100, 200, 200]], 2.0)
        self.assertEqual(boxes[0], [50.0, 50.0, 250.0, 250.0])

    def test_clips_at_image_edge(self):
        boxes = scale_boxes([[10, 10, 110, 110]], 2.0)
        self.assertEqual(boxes[0], [0.0, 0.0, 160.0, 160.0])

    def test_box_scale_can_rescue_small_box(self):
        gold = _label([[100, 100, 200, 200]])
        pred = _label([[120, 120, 180, 180]])  # IoU 0.36 < 0.5
        rows = [{"response": pred, "labels": gold}]
        raw = evaluate_rows(rows, coord="pixel")
        calibrated = evaluate_rows(rows, coord="pixel", box_scale=1.3)
        self.assertEqual(raw["localization"]["f1@0.5"], 0.0)
        self.assertEqual(calibrated["localization"]["f1@0.5"], 1.0)
        self.assertEqual(calibrated["meta"]["box_scale"], 1.3)


class MatchingTest(unittest.TestCase):
    def test_greedy_one_to_one(self):
        pred = [[0, 0, 100, 100], [90, 90, 200, 200]]
        gold = [[0, 0, 100, 100], [95, 95, 200, 200]]
        matched = greedy_match(pred, gold)
        self.assertEqual(len(matched), 2)
        self.assertEqual({p for p, _, _ in matched}, {0, 1})
        self.assertEqual({g for _, g, _ in matched}, {0, 1})

    def test_multi_box_counts_not_first_only(self):
        """Both boxes must be counted; recall is not based on the first box only."""
        pred = [[0, 0, 50, 50], [200, 200, 260, 260]]
        gold = [[0, 0, 50, 50], [200, 200, 260, 260]]
        metrics = localization_metrics(pred, gold)
        self.assertEqual(metrics["recall@0.5"], 1.0)
        self.assertEqual(metrics["precision@0.5"], 1.0)
        self.assertEqual(metrics["map50_95"], 1.0)

    def test_extra_prediction_lowers_precision(self):
        pred = [[0, 0, 50, 50], [200, 200, 260, 260]]
        gold = [[0, 0, 50, 50]]
        metrics = localization_metrics(pred, gold)
        self.assertEqual(metrics["recall@0.5"], 1.0)
        self.assertEqual(metrics["precision@0.5"], 0.5)


class DiagnosisTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(
            normalize_diagnosis("  Multiple intra-cranial MENINGIOMAS. "),
            "multiple intra cranial meningioma",
        )

    def test_strict_vs_lenient(self):
        self.assertTrue(diagnosis_match("Meningioma", "meningioma", lenient=False))
        self.assertTrue(diagnosis_match("Multiple intra-cranial meningiomas",
                                        "Meningioma"))
        self.assertFalse(
            diagnosis_match("Multiple intra-cranial meningiomas", "Meningioma",
                            lenient=False)
        )
        self.assertFalse(diagnosis_match("Glioma", "Meningioma"))

    def test_synonyms(self):
        self.assertTrue(diagnosis_match("haemorrhage", "hemorrhage", lenient=False))
        self.assertTrue(diagnosis_match("brain metastases", "brain metastasis",
                                        lenient=False))

    def test_bilingual_answer_compares_on_english_tokens(self):
        pred = "小脑星形细胞瘤（Cerebellar Astrocytoma），特别是毛细胞型星形细胞瘤（Pilocytic Astrocytoma）"
        self.assertTrue(diagnosis_match(pred, "Pilocytic astrocytoma"))
        self.assertFalse(diagnosis_match(pred, "Meningioma"))

    def test_pure_chinese_answer_does_not_match_english_gold(self):
        self.assertFalse(diagnosis_match("毛细胞型星形细胞瘤", "Pilocytic astrocytoma"))


class DiagnosisKeywordTest(unittest.TestCase):
    def test_alias_file_covers_test_labels(self):
        aliases = load_diagnosis_aliases()
        self.assertGreaterEqual(len(aliases), 30)
        self.assertIn("pilocytic astrocytoma", aliases)

    def test_keyword_hits_paraphrase(self):
        self.assertTrue(
            diagnosis_keyword_match(
                "Cerebellar cystic pilocytic astrocytoma", "Pilocytic astrocytoma"
            )
        )
        self.assertTrue(
            diagnosis_keyword_match(
                "Leigh syndrome", "Leigh's disease (Mitochondrial disorder)"
            )
        )
        self.assertTrue(
            diagnosis_keyword_match(
                "Craniopharyngioma",
                "Giant Adamantinomatous craniopharyngioma with obstructive mild hydrocephalus",
            )
        )
        self.assertTrue(
            diagnosis_keyword_match(
                "Multiple system atrophy (MSA-C)",
                "Multiple System Atrophy - Cerebellar type",
            )
        )

    def test_keyword_rejects_shared_generic_words(self):
        # 共享 encephalopathy 但实体不同，不能算命中
        self.assertFalse(
            diagnosis_keyword_match(
                "Vitamin B12 deficiency encephalopathy", "Wernicke Encephalopathy"
            )
        )
        # 共享 midbrain 但实体不同
        self.assertFalse(
            diagnosis_keyword_match(
                "Midbrain hemorrhage",
                "Bithalamic and cranial-midbrain infarction from occlusion of the Percheron artery",
            )
        )
        self.assertFalse(
            diagnosis_keyword_match(
                "Recurrent meningioma", "Multiple intra-cranial meningiomas"
            )
        )

    def test_unknown_label_falls_back_to_lenient(self):
        self.assertTrue(
            diagnosis_keyword_match("Meningioma", "Meningioma NOS")
        )
        self.assertFalse(diagnosis_keyword_match("", "Pilocytic astrocytoma"))


class CaptionTest(unittest.TestCase):
    def test_bleu_perfect(self):
        self.assertAlmostEqual(corpus_bleu(["a b c d"], ["a b c d"]), 1.0, places=6)

    def test_bleu_unrelated(self):
        self.assertLess(corpus_bleu(["x y z"], ["a b c"]), 0.2)


class EvaluateRowsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        rows = []
        for i in range(3):
            text = _label([[10 + i, 20, 100, 120]])
            rows.append({"response": text, "labels": text, "messages": []})
        report = evaluate_rows(rows)
        self.assertEqual(report["meta"]["coord_mode"], "pixel")
        self.assertAlmostEqual(report["diagnosis"]["top1_lenient"], 1.0)
        self.assertAlmostEqual(report["diagnosis"]["top1_keyword"], 1.0)
        self.assertAlmostEqual(report["cases"]["top1_keyword"], 1.0)
        self.assertAlmostEqual(report["localization"]["mean_iou"], 1.0)
        self.assertAlmostEqual(report["caption"]["f1"], 1.0)
        self.assertAlmostEqual(report["caption"]["bleu4"], 1.0)
        self.assertAlmostEqual(report["cases"]["top1_lenient"], 1.0)

    def test_norm1000_predictions_still_match(self):
        gold_text = _label([[300, 20, 400, 120]])
        pred_text = _label([[625, 41.7, 833.3, 250]])
        report = evaluate_rows([{"response": pred_text, "labels": gold_text}])
        self.assertEqual(report["meta"]["coord_mode"], "norm1000")
        self.assertGreaterEqual(report["localization"]["mean_iou"], 0.99)

    def test_missing_sections_counted(self):
        gold_text = _label([[10, 20, 100, 120]])
        report = evaluate_rows([{"response": "no structured output", "labels": gold_text}])
        self.assertEqual(report["parsing"]["no_box_pred"], 1)
        self.assertEqual(report["diagnosis"]["top1_lenient"], 0.0)

    def test_empty_response_does_not_fall_back_to_label(self):
        gold_text = _label([[10, 20, 100, 120]])
        report = evaluate_rows(
            [
                {
                    "response": "",
                    "labels": gold_text,
                    "messages": [{"role": "assistant", "content": gold_text}],
                }
            ]
        )
        self.assertEqual(report["parsing"]["empty_pred"], 1)
        self.assertEqual(report["diagnosis"]["top1_lenient"], 0.0)

    def test_ambiguous_coord_range_is_flagged(self):
        gold_text = _label([[100, 20, 300, 120]])
        pred_text = _label([[200, 40, 640, 240]])
        report = evaluate_rows([{"response": pred_text, "labels": gold_text}])
        self.assertEqual(report["meta"]["coord_mode"], "norm1000")
        self.assertTrue(report["meta"]["coord_ambiguous"])

    def test_gold_rows_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate_rows([{"response": "x"}], [{"labels": "y"}, {"labels": "z"}])

    def test_case_grouping(self):
        gold = _label([[10, 20, 100, 120]])
        rows = [
            {"response": gold, "labels": gold,
             "messages": [{"image": ["/x/case0001_001.png"]}]},
            {"response": gold, "labels": gold,
             "messages": [{"image": ["/x/case0001_002.png"]}]},
            {"response": gold, "labels": gold,
             "messages": [{"image": ["/x/case0002_001.png"]}]},
        ]
        report = evaluate_rows(rows)
        self.assertEqual(report["cases"]["cases"], 2.0)

    def test_case_grouping_with_row_level_images(self):
        gold = _label([[10, 20, 100, 120]])
        rows = [
            {"response": gold, "labels": gold,
             "images": ["/x/case0001_001.png"],
             "messages": [{"role": "user", "content": "q"}]},
            {"response": gold, "labels": gold,
             "images": ["/x/case0001_002.png"],
             "messages": [{"role": "user", "content": "q"}]},
            {"response": gold, "labels": gold,
             "images": ["/x/case0002_001.png"],
             "messages": [{"role": "user", "content": "q"}]},
        ]
        report = evaluate_rows(rows)
        self.assertEqual(report["cases"]["cases"], 2.0)


class ReadRowsTest(unittest.TestCase):
    def test_skips_swift_preamble(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.jsonl"
            path.write_text(
                "run sh: `python infer.py ...`\n"
                '{"index": 0, "response": "x"}\n'
                "not json\n",
                encoding="utf-8",
            )
            rows = read_rows(path)
            self.assertEqual([row["index"] for row in rows], [0])


if __name__ == "__main__":
    unittest.main()
