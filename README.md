Python KD-Tree for Points
=========================

A simple and decently performant KD-Tree in Python.

Just about 60 lines of code excluding comments.

It's so simple that you can just copy and paste, or translate to other languages!   
Your teacher will assume that you are a good student who coded it from scratch.

Why?
----

No external dependencies like numpy, scipy, etc.

Supports points that are array-like: lists, arrays, numpy arrays.

Just star this project if you find it helpful... so others can know it's better than those long winded kd-tree codes. ;)

Requirements
------------

Python 2.x or 3.x

Dependencies
------------

None

Notes
-----

Creation of the KD-Tree isn't strictly O(n log (n)), but is similar O(n log (n)) in practice.   
It abuses Python's native sort (TimSort) which is O(n) for nearly sorted lists.   

Adding too many points relative to the number of points in the tree can degrade performance.   
If you are adding many new points into the tree, it is better to re-create the tree.

License
-------

CC0