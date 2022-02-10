class KDTree(object):
    
    """
    A super short KD-Tree for points...
    so concise that you can copypasta into your homework 
    without arousing suspicion.

    This implementation only supports Euclidean distance. 

    The points can be any array-like type, e.g: 
        lists, tuples, numpy arrays.

    Usage:
    1. Make the KD-Tree:
        `kd_tree = KDTree(points, dim)`
    2. You can then use `get_knn` for k nearest neighbors or 
       `get_nearest` for the nearest neighbor

    points are be a list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
    """
    def __init__(self, points, dim, dist_sq_func=None):
        """Makes the KD-Tree for fast lookup.

        Parameters
        ----------
        points : list<list or array>
            A list of points.
        dim : int 
            The dimension of the points. 
        dist_sq_func : function(point, point), optional
            A function that returns the squared Euclidean distance
            between the two points. 
        """

        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 
                for i, x in enumerate(a))
                
        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), 
                    points[m]]
            if len(points) == 1:
                return [None, None, points[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][i] - point[i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        import heapq
        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq, 
                        heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2] 
                    for h in sorted(heap)][::-1]

        self._add_point = add_point
        self._get_knn = get_knn 
        self._root = make(points)
        
    def add_point(self, point):
        """Adds a point to the kd-tree
        
        Parameters
        ----------
        point : array-like
            The point.
        """
        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):
        """Returns k nearest neighbors

        Parameters
        ----------
        point : array-like
            The point.
        k: int 
            The number of nearest neighbors.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distances.

        Returns
        -------
        list<array-like>
            The nearest neighbors. 
            If `return_dist_sq` is true, the return will be:
                [(dist_sq, point), ...]
            else:
                [point, ...]
        """
        return self._get_knn(self._root, point, k, return_dist_sq, [])

    def get_nearest(self, point, return_dist_sq=True):
        """Returns the nearest neighbor.

        Parameters
        ----------
        point : array-like
            The point.
        return_dist_sq : boolean
            Whether to return the squared Euclidean distance.

        Returns
        -------
        array-like
            The nearest neighbor. 
            If the tree is empty, returns `None`.
            If `return_dist_sq` is true, the return will be:
                (dist_sq, point)
            else:
                point
        """
        l = self._get_knn(self._root, point, 1, return_dist_sq, [])
        return l[0] if len(l) else None


if __name__ == '__main__':

    import random, cProfile

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
    additional_points = [rand_point(dim) for x in range(50)]
    query_points = [rand_point(dim) for x in range(100)]

    kd_tree_results = []
    naive_results = []

    def test_and_bench_kd_tree():
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

    print("Testing and benchmarking KDTree...")
    cProfile.run("test_and_bench_kd_tree()")
    print("Testing and benchmarking naive version...")
    cProfile.run("test_and_bench_naive()")

    print("Is the result same as naive version?: {}"
        .format(kd_tree_results == naive_results))
