"""
BK-Tree (Burkhard-Keller Tree)

=== THEORY ===

A BK-tree is a metric tree designed for nearest-neighbour search in
discrete metric spaces.  Here we use it with Levenshtein distance to
find all dictionary words within `max_edit_distance` of a query word.

The key insight that makes BK-trees work is the TRIANGLE INEQUALITY:

   d(a, c) ≤ d(a, b) + d(b, c)

So if we are looking for words within distance k of query q, and the
current node n has distance d(n, q) = r, then any matching descendant
must have distance in the range [r−k, r+k] from n.  This prunes large
portions of the tree.

=== STRUCTURE ===

  BKTree Node:
    word      — the stored word
    children  — dict { edge_distance → child_node }

Insert("cat"):
  root = Node("cat")

Insert("bat"):
  d("cat", "bat") = 1  →  root.children[1] = Node("bat")

Insert("hat"):
  d("cat", "hat") = 1  → follow edge 1 to "bat"
  d("bat", "hat") = 1  → bat.children[1] = Node("hat")

Insert("have"):
  d("cat", "have") = 4  → root.children[4] = Node("have")

Search(q="can", k=1):
  At root("cat"):  d("cat","can") = 1 ≤ k  → candidate
    Search edges [1-1, 1+1] = [0, 2]: check children with edge 1
    At "bat":  d("bat","can") = 2 > k  → skip
  Result: ["cat"]

=== COMPLEXITY ===

  Build (N words):   O(N · avg_depth · dist_time)  ≈ O(N log N)
  Query:             O(N^0.36)  empirically on English dictionaries
                     Degenerates to O(N) for large k

=== AT GOOGLE SCALE ===

Google uses neural spell correction (Seq2Seq / T5) trained on
query-correction pairs from click logs.  BK-trees are used in
lightweight on-device correction, keyboards (e.g. Android/iOS autocorrect),
and as a fast first pass before a more expensive model.
"""

import logging
from typing import Optional

from app.spellcheck.levenshtein import levenshtein_distance

logger = logging.getLogger(__name__)


class BKNode:
    __slots__ = ("word", "children")

    def __init__(self, word: str) -> None:
        self.word = word
        self.children: dict[int, "BKNode"] = {}


class BKTree:
    """
    BK-tree for fast approximate string matching.

    Example
    -------
    >>> tree = BKTree()
    >>> for w in ["python", "java", "jawa", "pythn"]:
    ...     tree.insert(w)
    >>> tree.search("pyton", max_distance=1)
    [('python', 1)]
    """

    def __init__(self) -> None:
        self._root: Optional[BKNode] = None
        self._size: int = 0

    def insert(self, word: str) -> None:
        """Insert a word into the BK-tree."""
        word = word.lower().strip()
        if not word:
            return

        if self._root is None:
            self._root = BKNode(word)
            self._size += 1
            return

        node = self._root
        while True:
            dist = levenshtein_distance(word, node.word)
            if dist == 0:
                return   # already present
            if dist not in node.children:
                node.children[dist] = BKNode(word)
                self._size += 1
                return
            node = node.children[dist]

    def search(self, query: str, max_distance: int = 2) -> list[tuple[str, int]]:
        """
        Return all (word, distance) pairs where distance ≤ max_distance.
        Results are sorted by distance ascending.
        """
        query = query.lower().strip()
        if not query or self._root is None:
            return []

        results: list[tuple[str, int]] = []
        stack = [self._root]

        while stack:
            node = stack.pop()
            dist = levenshtein_distance(query, node.word, max_distance=max_distance + 1)

            if dist <= max_distance:
                results.append((node.word, dist))

            # Triangle inequality pruning
            lo = dist - max_distance
            hi = dist + max_distance
            for edge_dist, child in node.children.items():
                if lo <= edge_dist <= hi:
                    stack.append(child)

        results.sort(key=lambda x: x[1])
        return results

    def bulk_insert(self, words: list[str]) -> None:
        """Insert many words at once."""
        for word in words:
            self.insert(word)

    @property
    def size(self) -> int:
        return self._size
