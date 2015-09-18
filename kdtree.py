"""
A damm short KD-Tree for points...
so concise that you can copypasta into your homework without arousing suspicion.


Usage:
1. Use make_kd_tree to create the kd
2. You can then use `get_knn` for k nearest neighbors or 
   `get_nearest` for the nearest neighbor

points is list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
or whatever that can be accessed like a list.
"""

# Makes the KD-Tree far fast lookup
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return (
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half])
    elif len(points) == 1:
        return (None, None, points[0])

# K nearest neighbors. The heap is a bounded priority queue.
def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        get_knn(kd_node[dx < 0], point, k, dim, dist_func, return_distances, i, heap)
        if dx * dx < -heap[0][0]: # -heap[0][0] is the largest distance in the heap
            get_knn(kd_node[dx >= 0], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]

# For the closest neighbor
def get_nearest(kd_node, point, dim, dist_func, return_distances=True, i=0, best=None):
    if kd_node:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if not best:
            best = [dist, kd_node[2]]
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        get_nearest(kd_node[dx < 0], point, dim, dist_func, return_distances, i, best)
        if dx * dx < best[0]:
            get_nearest(kd_node[dx >= 0], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]



"""
If you want to attach other properties to your points, 
you can use this class or subclass it.

Usage:

point = PointContainer([1,2,3])
point.label = True  
print point         # [1,2,3]
print point.label   # True 
"""
class PointContainer(list):
    def __new__(self, value, name = None, values = None):
        s = super(PointContainer, self).__new__(self, value)
        return s


"""
Below is all the testing code
"""

import random, cProfile


def puts(l):
    for x in l:
        print x


def get_knn_naive(points, point, k, dist_func, return_distances=True):
    neighbors = []
    for i, pp in enumerate(points):
        dist = dist_func(point, pp)
        neighbors.append((dist, pp))
    neighbors = sorted(neighbors)[:k]
    return neighbors if return_distances else [n[1] for n in neighbors]

dim = 3

def rand_point(dim):
    return [random.uniform(-1, 1) for d in xrange(dim)]

def dist_sq(a, b, dim):
    return sum((a[i] - b[i]) ** 2 for i in xrange(dim))

def dist_sq_dim(a, b):
    return dist_sq(a, b, dim)


points = [PointContainer(rand_point(dim)) for x in xrange(5000)]
#points = [rand_point(dim) for x in xrange(5000)]
test = [rand_point(dim) for x in xrange(1000)]
result1 = []
result2 = []


def bench1():
    kd_tree = make_kd_tree(points, dim)
    result1.append(tuple(get_knn(kd_tree, [0] * dim, 8, dim, dist_sq_dim)))
    for t in test:
        result1.append(tuple(get_knn(kd_tree, t, 8, dim, dist_sq_dim)))


def bench2():
    result2.append(tuple(get_knn_naive(points, [0] * dim, 8, dist_sq_dim)))
    for t in test:
        result2.append(tuple(get_knn_naive(points, t, 8, dist_sq_dim)))

cProfile.run("bench1()")
cProfile.run("bench2()")

puts(result1[0])
print
puts(result2[0])
print

print "Is the result same as naive version?: %s" % (result1 == result2)

print
kd_tree = make_kd_tree(points, dim)

print get_nearest(kd_tree, [0] * dim, dim, dist_sq_dim)

"""
You can also define the distance function inline, like:

print get_nearest(kd_tree, [0] * dim, dim, lambda a,b: dist_sq(a, b, dim))
print get_nearest(kd_tree, [0] * dim, dim, lambda a,b: sum((a[i] - b[i]) ** 2 for i in xrange(dim)))
"""




