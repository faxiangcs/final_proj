"""Protect the two resampling units and per-image weighting used by the paper."""
import unittest
from unittest.mock import Mock, patch
from scripts.run_refinement import paired_seed_case_ci


class SeedCaseBootstrapTest(unittest.TestCase):
    def test_keeps_image_weights_in_unequal_size_cases(self):
        rng = Mock()
        rng.randrange.return_value = 0
        rng.choice.side_effect = ['A','B']
        with patch('scripts.run_refinement.random.Random', return_value=rng):
            interval = paired_seed_case_ci([[0,0,1]], ['A','A','B'], n_boot=1)
        self.assertEqual(interval, [1/3,1/3])

    def test_resamples_seeds_instead_of_averaging_them_first(self):
        rng = Mock()
        rng.randrange.return_value = 1
        rng.choice.side_effect = ['A','B']
        with patch('scripts.run_refinement.random.Random', return_value=rng):
            interval = paired_seed_case_ci([[0,0],[1,1]], ['A','B'], n_boot=1)
        self.assertEqual(interval, [1,1])

    def test_constant_delta_and_reproducibility(self):
        values = [[.25,.25,.25]]*3
        self.assertEqual(paired_seed_case_ci(values, ['A','A','B'], n_boot=20), [.25,.25])
        values = [[0,1,0],[1,0,0],[1,1,1]]
        self.assertEqual(paired_seed_case_ci(values, ['A','A','B'], n_boot=30),
                         paired_seed_case_ci(values, ['A','A','B'], n_boot=30))

    def test_rejects_misaligned_images(self):
        with self.assertRaises(ValueError):
            paired_seed_case_ci([[1]], ['A','B'])


if __name__ == '__main__':
    unittest.main()
