import random
import unittest

from src.algorithms import (
    Item, exact_tsp_cost, greedy_knapsack, mst_preorder_tour, nearest_neighbor_tour,
    random_tour, tour_cost, two_opt,
)


class AlgorithmTests(unittest.TestCase):
    def test_lightest_knapsack_can_be_worse(self):
        items = [Item("A", 2, 3), Item("B", 3, 5), Item("C", 3, 5)]
        self.assertEqual(greedy_knapsack(items, 6, "value_weight")[2], 10)
        self.assertEqual(greedy_knapsack(items, 6, "lightest")[2], 8)

    def test_two_opt_does_not_worsen_tour(self):
        points = [(0, 0), (1, 1), (0, 1), (1, 0), (.5, .4)]
        initial = [0, 2, 1, 3, 4]
        result, improvements, _ = two_opt(initial, points)
        self.assertLessEqual(tour_cost(result, points), tour_cost(initial, points) + 1e-9)
        self.assertGreaterEqual(improvements, 0)

    def test_mst_tour_and_exact_cost(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tour = mst_preorder_tour(points)
        self.assertEqual(set(tour), set(range(4)))
        self.assertGreaterEqual(tour_cost(tour, points) + 1e-9, exact_tsp_cost(points))

    def test_nearest_neighbor_is_permutation(self):
        points = [(random.Random(i).random(), random.Random(i + 10).random()) for i in range(8)]
        self.assertEqual(set(nearest_neighbor_tour(points, 3)), set(range(8)))
        self.assertEqual(set(random_tour(8, random.Random(2))), set(range(8)))


if __name__ == "__main__":
    unittest.main()
