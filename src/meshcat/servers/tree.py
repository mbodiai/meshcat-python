from __future__ import absolute_import, division, print_function

from collections import defaultdict
from typing import List, Optional, TypeVar, Generic
from collections.abc import Iterator

# NOTE: We purposely keep the dynamic "infinite tree" behaviour of the original
# implementation (each missing key automatically creates a new ``TreeNode``)
# while also adding *static* type information so that modern type checkers such
# as Pyright/Mypy understand that ``object``, ``transform`` … attributes exist
# and can hold arbitrary binary payloads (``bytes``).

T = TypeVar("T", bound="TreeNode")


class TreeNode(defaultdict[str, "TreeNode"], Generic[T]):
    """A recursive defaultdict used by MeshCat to represent the scene graph.

    The class intentionally mirrors the original untyped implementation while
    annotating public attributes so that static analysers stop complaining when
    we assign ``bytes`` to them inside *zmqserver.py*.
    """

    __slots__ = ["object", "transform", "properties", "animation"]

    # Public payload slots – all of them store raw binary packets that the
    # front-end can parse (protobuf, msgpack, …).  ``None`` denotes *unset*.
    object: Optional[bytes]
    transform: Optional[bytes]
    properties: List[bytes]
    animation: Optional[bytes]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object = None
        self.properties = []
        self.transform = None
        self.animation = None

def SceneTree() -> TreeNode:  # type: ignore[valid-type]
    """Factory function used for historical reasons.

    The original JavaScript viewer expected a *callable* rather than a *class*
    so we keep the public symbol unchanged.  Returning a new ``TreeNode`` keeps
    runtime behaviour intact while the explicit return type placates
    type-checkers.
    """

    return TreeNode(SceneTree)  # type: ignore[arg-type]

def walk(tree: TreeNode) -> Iterator[TreeNode]:
    yield tree
    for v in tree.values():
        for t in walk(v):  # could use `yield from` if we didn't need python2
            yield t

def find_node(tree: TreeNode, path: list[str]) -> TreeNode:
    if len(path) == 0:
        return tree
    else:
        return find_node(tree[path[0]], path[1:])
