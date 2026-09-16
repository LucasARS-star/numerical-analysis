import math
import unittest

from implicit_euler import implicit_euler


class ImplicitEulerTests(unittest.TestCase):
    def test_linear_decay_matches_backward_euler_formula(self):
        result = implicit_euler(
            lambda _t, y: -15.0 * y,
            0.0,
            1.0,
            0.1,
            2,
            jacobian=lambda _t, _y: -15.0,
        )

        self.assertEqual(result.times, [0.0, 0.1, 0.2])
        self.assertTrue(math.isclose(result.values[-1], 1.0 / 2.5**2))

    def test_uses_numerical_jacobian_when_one_is_not_supplied(self):
        result = implicit_euler(lambda _t, y: -2.0 * y, 0.0, 3.0, 0.25, 1)

        self.assertTrue(math.isclose(result.values[-1], 2.0, abs_tol=1e-10))

    def test_rejects_invalid_step_size(self):
        with self.assertRaises(ValueError):
            implicit_euler(lambda _t, y: y, 0.0, 1.0, 0.0, 1)


if __name__ == "__main__":
    unittest.main()
