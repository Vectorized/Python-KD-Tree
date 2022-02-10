import unittest
import random
import cProfile
from kd_tree import *

class KDTreeUnitTest(unittest.TestCase):

    def test_all(self):

        dim = 3

        def dist_sq_func(a, b):
            return sum((x - b[i]) ** 2 for i, x in enumerate(a))

        def get_knn_naive(points, point, k, return_dist_sq=True):
            neighbors = []
            for i, pp in enumerate(points):
                dist_sq = dist_sq_func(point, pp)
                neighbors.append((dist_sq, pp))
            neighbors = sorted(neighbors)[:k]
            return neighbors if return_dist_sq else [n[1] for n in neighbors]

        def get_nearest_naive(points, point, return_dist_sq=True):
            nearest = min(points, key=lambda p:dist_sq_func(p, point))
            if return_dist_sq:
                return (dist_sq_func(nearest, point), nearest) 
            return nearest

        def rand_point(dim):
            return [random.uniform(-1, 1) for d in range(dim)]

        points = [rand_point(dim) for x in range(10000)]
        additional_points = [rand_point(dim) for x in range(100)]
        query_points = [rand_point(dim) for x in range(100)]

        kd_tree_results = []
        naive_results = []
        
        global test_and_bench_kd_tree
        global test_and_bench_naive

        def test_and_bench_kd_tree():
            global kd_tree
            kd_tree = KDTree(points, dim)
            for point in additional_points:
                kd_tree.add_point(point)
            kd_tree_results.append(tuple(kd_tree.get_knn([0] * dim, 8)))
            for t in query_points:
                kd_tree_results.append(tuple(kd_tree.get_knn(t, 8)))
            for t in query_points:
                kd_tree_results.append(tuple(kd_tree.get_nearest(t)))

        def test_and_bench_naive():
            all_points = points + additional_points
            naive_results.append(tuple(get_knn_naive(all_points, [0] * dim, 8)))
            for t in query_points:
                naive_results.append(tuple(get_knn_naive(all_points, t, 8)))
            for t in query_points:
                naive_results.append(tuple(get_nearest_naive(all_points, t)))

        print("Running KDTree...")
        cProfile.run("test_and_bench_kd_tree()")
        
        print("Running naive version...")
        cProfile.run("test_and_bench_naive()")

        print("Query results same as naive version?: {}"
            .format(kd_tree_results == naive_results))
        
        self.assertEqual(kd_tree_results, naive_results, 
            "Query results mismatch")
        
        self.assertEqual(len(list(kd_tree)), len(points) + len(additional_points), 
            "Number of points from iterator mismatch")

if __name__ == '__main__':
    unittest.main()
