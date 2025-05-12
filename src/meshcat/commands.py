from meshcat.animation import Animation
from .geometry import Geometry, Object, Mesh, MeshPhongMaterial, OrthographicCamera, PerspectiveCamera, PointsMaterial, Points, TextTexture
from .path import Path
from typing import Any

class SetObject:
    __slots__ = ["object", "path"]
    def __init__(self, geometry_or_object, material=None, path=None):
        if isinstance(geometry_or_object, Object):
            if material is not None:
                raise(ValueError("Please supply either an Object OR a Geometry and a Material"))
            self.object = geometry_or_object
        elif isinstance(geometry_or_object, (OrthographicCamera, PerspectiveCamera)):
            self.object = geometry_or_object
        else:
            if material is None:
                material = MeshPhongMaterial()
            if isinstance(material, PointsMaterial):
                self.object = Points(geometry_or_object, material)
            else:
                self.object = Mesh(geometry_or_object, material)
        if path is not None:
            self.path = path
        else:
            self.path = Path()

    def lower(self):
        return {
            "type": "set_object",
            "object": self.object.lower(),
            "path": self.path.lower()
        }


class SetTransform:
    __slots__ = ["matrix", "path"]
    def __init__(self, matrix, path):
        self.matrix = matrix
        self.path = path

    def lower(self):
        return {
            u"type": u"set_transform",
            u"path": self.path.lower(),
            u"matrix": list(self.matrix.T.flatten())
        }


class SetCamTarget:
    """Set the camera target point."""
    __slots__ = ["value"]
    def __init__(self, pos):
        self.value = pos

    def lower(self):
        return {
            u"type": "set_target",
            u"path": "",
            u"value": list(self.value)
        }


class CaptureImage:

    def __init__(self, xres: int | None = None, yres: int | None = None):
        self.xres = xres
        self.yres = yres

    def lower(self) -> dict[str, int|str]:
        data: dict[str, int|str] = {
            "type": "capture_image"
        }
        if self.xres:
            data["xres"] = self.xres
        if self.yres:
            data["yres"] = self.yres
        return data


class Delete:
    __slots__ = ["path"]
    def __init__(self, path):
        self.path = path

    def lower(self):
        return {
            "type": "delete",
            "path": self.path.lower()
        }

class SetProperty:
    __slots__ = ["path", "key", "value"]
    def __init__(self, key: str, value: Any, path: str):
        self.key = key
        self.value = value
        self.path = path

    def lower(self):
        return {
            "type": "set_property",
            "path": self.path.lower(),
            "property": self.key.lower(),
            "value": self.value
        }

class SetAnimation:
    __slots__ = ["animation", "play", "repetitions"]

    def __init__(self, animation: "Animation", play: bool = True, repetitions: int = 1):
        self.animation = animation
        self.play = play
        self.repetitions = repetitions

    def lower(self):
        return {
            "type": "set_animation",
            "animations": self.animation.lower(),
            "options": {
                "play": self.play,
                "repetitions": self.repetitions
            },
            "path": ""
        }
