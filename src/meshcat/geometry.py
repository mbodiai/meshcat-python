import base64
import uuid
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import umsgpack
import numpy as np
from meshcat.transformations import identity_matrix


class SceneElement(object):
    _type: str | None = None
    field: str | None = None

    def __init__(self):
        self.uuid = str(uuid.uuid1())
    

class ReferenceSceneElement(SceneElement):
    field: str # type: ignore
    def lower_in_object(self, object_data:dict[str,list]):
        object_data.setdefault(self.field, []).append(self.lower(object_data))
        return self.uuid

    def lower(self, object_data:dict) -> dict:
        """
        Placeholder lower method. Subclasses should implement their specific
        logic for converting the object to a dictionary representation.
        """
        raise NotImplementedError("Subclasses must implement the lower() method.")


class Geometry(ReferenceSceneElement):
    field = "geometries"


    def intrinsic_transform(self):
        return identity_matrix()


class Material(ReferenceSceneElement):
    field = "materials"


class Texture(ReferenceSceneElement):
    field = "textures"


class Image(ReferenceSceneElement):
    field = "images"


class Box(Geometry):
    def __init__(self, lengths):
        super().__init__()
        self.lengths = lengths

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "BoxGeometry",
            "width": self.lengths[0],
            "height": self.lengths[1],
            "depth": self.lengths[2]
        }


class Sphere(Geometry):
    def __init__(self, radius:float):
        super().__init__()
        self.radius = radius

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "SphereGeometry",
            "radius": self.radius,
            "widthSegments" : 20,
            "heightSegments" : 20
        }


class Ellipsoid(Sphere):
    """
    An Ellipsoid is treated as a Sphere of unit radius, with an affine
    transformation applied to distort it into the ellipsoidal shape
    """
    def __init__(self, radii:ArrayLike):
        super().__init__(1.0)
        self.radii = np.asarray(radii, dtype=np.float32)

    def intrinsic_transform(self):
        return np.diag(np.hstack((self.radii, 1.0)))


class Plane(Geometry):

    def __init__(self, width:float=1.0, height:float=1.0, widthSegments:int=1, heightSegments:int=1):
        super().__init__()
        self.width = width
        self.height = height
        self.widthSegments = widthSegments
        self.heightSegments = heightSegments

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "PlaneGeometry",
            "width": self.width,
            "height": self.height,
            "widthSegments": self.widthSegments,
            "heightSegments": self.heightSegments,
        }



class Cylinder(Geometry):
    """A cylinder of the given height and radius. By Three.js convention, the axis of
    rotational symmetry is aligned with the y-axis.
    """
    def __init__(self, height, radius=1.0, radiusTop=None, radiusBottom=None):
        super().__init__()
        if radiusTop is not None and radiusBottom is not None:
            self.radiusTop = radiusTop
            self.radiusBottom = radiusBottom
        else:
            self.radiusTop = radius
            self.radiusBottom = radius
        self.height = height
        self.radialSegments = 50

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "CylinderGeometry",
            "radiusTop": self.radiusTop,
            "radiusBottom": self.radiusBottom,
            "height": self.height,
            "radialSegments": self.radialSegments
        }


class GenericMaterial(Material):
    _type:str
    def __init__(self, color:int=0xffffff, reflectivity:float=0.5, map:Texture|None=None,
                 side:int=2, transparent:bool|None=None, opacity:float=1.0,
                 linewidth:float=1.0,
                 wireframe:bool=False,
                 wireframeLinewidth:float=1.0,
                 vertexColors:bool=False,
                 **kwargs):
        super().__init__()
        self.color = color
        self.reflectivity = reflectivity
        self.map = map
        self.side = side
        self.transparent = transparent
        self.opacity = opacity
        self.linewidth = linewidth
        self.wireframe = wireframe
        self.wireframeLinewidth = wireframeLinewidth
        self.vertexColors = vertexColors
        self.properties = kwargs

    def lower(self, object_data):
        # Three.js allows a material to have an opacity which is != 1,
        # but to still be non-transparent, in which case the opacity only
        # serves to desaturate the material's color. That's a pretty odd
        # combination of things to want, so by default we juse use the
        # opacity value to decide whether to set transparent to True or
        # False.
        if self.transparent is None:
            transparent = bool(self.opacity != 1)
        else:
            transparent = self.transparent
        data = {
            "uuid": self.uuid,
            "type": self._type,
            "color": self.color,
            "reflectivity": self.reflectivity,
            "side": self.side,
            "transparent": transparent,
            "opacity": self.opacity,
            "linewidth": self.linewidth,
            "wireframe": bool(self.wireframe),
            "wireframeLinewidth": self.wireframeLinewidth,
            "vertexColors": (2 if self.vertexColors else 0),  # three.js wants an enum
        }
        data.update(self.properties)
        if self.map is not None:
            data["map"] = self.map.lower_in_object(object_data)
        return data


class MeshBasicMaterial(GenericMaterial):
    _type="MeshBasicMaterial"


class MeshPhongMaterial(GenericMaterial):
    _type="MeshPhongMaterial"


class MeshLambertMaterial(GenericMaterial):
    _type="MeshLambertMaterial"


class MeshToonMaterial(GenericMaterial):
    _type="MeshToonMaterial"


class LineBasicMaterial(GenericMaterial):
    _type="LineBasicMaterial"


class PngImage(Image):
    def __init__(self, data):
        super().__init__()
        self.data = data

    @staticmethod
    def from_file(fname):
        with open(fname, "rb") as f:
            return PngImage(f.read())

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "url": str("data:image/png;base64," + base64.b64encode(self.data).decode('ascii'))
        }


class TextTexture(Texture):
    def __init__(self, text, font_size=100, font_face='sans-serif'):
        super().__init__()
        self.text = text
        # font_size will be passed to the JS side as is; however if the
        # text width exceeds canvas width, font_size will be reduced.
        self.font_size = font_size
        self.font_face = font_face

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "_text",
            "text": self.text,
            "font_size": self.font_size,
            "font_face": self.font_face,
        }


class GenericTexture(Texture):
    def __init__(self, properties):
        super().__init__()
        self.properties = properties

    def lower(self, object_data):
        data = {"uuid": self.uuid}
        data.update(self.properties)
        if "image" in data:
            image_element = data["image"]
            # The original code implies image_element has lower_in_object.
            # If image_element is a string (UUID), this will error at runtime.
            # Keeping original logic as requested.
            data["image"] = image_element.lower_in_object(object_data) # type: ignore
        return data


class ImageTexture(Texture):
    def __init__(self, image, wrap=[1001, 1001], repeat=[1, 1], **kwargs):
        super().__init__()
        self.image = image
        self.wrap = wrap
        self.repeat = repeat
        self.properties = kwargs

    def lower(self, object_data):
        data = {
            "uuid": self.uuid,
            "wrap": self.wrap,
            "repeat": self.repeat,
            "image": self.image.lower_in_object(object_data)
        }
        data.update(self.properties)
        return data


class Object(SceneElement):
    _type: str # Mesh, Points, Line etc.

    def __init__(self, geometry:"Geometry", material:"Material|None"=None):
        super().__init__()
        self.geometry = geometry
        self.material = material or MeshLambertMaterial()

    def lower(self):
        data = {
            "metadata": {
                "version": 4.5,
                "type": "Object",
            },
            "geometries": [],
            "materials": [],
            "object": {
                "uuid": self.uuid,
                "type": self._type,
                "geometry": self.geometry.uuid,
                "material": self.material.uuid,
                "matrix": list(self.geometry.intrinsic_transform().flatten())
            }
        }
        self.geometry.lower_in_object(data)
        self.material.lower_in_object(data)
        return data


class Mesh(Object):
    _type = "Mesh"


class OrthographicCamera(SceneElement):
    def __init__(self, left, right, top, bottom, near, far, zoom=1):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.near = near
        self.far = far
        self.zoom = zoom

    def lower(self):
        data = {
            "object": {
                "uuid": self.uuid,
                "type": "OrthographicCamera",
                "left": self.left,
                "right": self.right,
                "top": self.top,
                "bottom": self.bottom,
                "near": self.near,
                "far": self.far,
                "zoom": self.zoom,
            }
        }
        return data

class PerspectiveCamera(SceneElement):
    """
    The PerspectiveCamera is the default camera used by the meshcat viewer. See
    https://threejs.org/docs/#api/en/cameras/PerspectiveCamera for more
    information.
    """
    def __init__(self, fov:float=50, aspect:float=1, near:float=0.1, far:float=2000,
                 zoom:float=1, filmGauge:float=35, filmOffset:float=0, focus:float=10):
        """
        fov   : Camera frustum vertical field of view, from bottom to top of view, in degrees. Default is 50.
        aspect: Camera frustum aspect ratio, usually the canvas width / canvas height. Default is 1 (square canvas).
        near  : Camera frustum near plane. Default is 0.1. The valid range is greater than 0 and less than the current
                value of the far plane. Note that, unlike for the OrthographicCamera, 0 is not a valid value for a
                PerspectiveCamera's near plane.
        far   : Camera frustum far plane. Default is 2000.
        zoom  : Gets or sets the zoom factor of the camera. Default is 1.
        filmGauge: Film size used for the larger axis. Default is 35 (millimeters). This parameter does not influence
                   the projection matrix unless .filmOffset is set to a nonzero value.
        filmOffset: Horizontal off-center offset in the same unit as .filmGauge. Default is 0.
        focus: Object distance used for stereoscopy and depth-of-field effects. This parameter does not influence
               the projection matrix unless a StereoCamera is being used. Default is 10.
        """
        super().__init__()
        self.fov = fov
        self.aspect = aspect
        self.far = far
        self.near = near
        self.zoom = zoom
        self.filmGauge = filmGauge
        self.filmOffset = filmOffset
        self.focus = focus

    def lower(self):
        data = {
            "object": {
                "uuid": self.uuid,
                "type": "PerspectiveCamera",
                "aspect": self.aspect,
                "far": self.far,
                "filmGauge": self.filmGauge,
                "filmOffset": self.filmOffset,
                "focus": self.focus,
                "fov": self.fov,
                "near": self.near,
                "zoom": self.zoom,
            }
        }
        return data

def item_size(array):
    if array.ndim == 1:
        return 1
    elif array.ndim == 2:
        return array.shape[0]
    else:
        raise ValueError("I can only pack 1- or 2-dimensional numpy arrays, but this one has {:d} dimensions".format(array.ndim))


def threejs_type(dtype):
    if dtype == np.uint8:
        return "Uint8Array", 0x12
    elif dtype == np.int32:
        return "Int32Array", 0x15
    elif dtype == np.uint32:
        return "Uint32Array", 0x16
    elif dtype == np.float32:
        return "Float32Array", 0x17
    else:
        raise ValueError("Unsupported datatype: " + str(dtype))


def pack_numpy_array(x:np.ndarray):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    typename, extcode = threejs_type(x.dtype)
    return {
        "itemSize": item_size(x),
        "type": typename,
        "array": umsgpack.Ext(extcode, x.tobytes('F')),
        "normalized": False
    }


def data_from_stream(stream):
    if isinstance(stream, BytesIO):
        data = stream.read().decode(encoding='utf-8')
    elif isinstance(stream, StringIO):
        data = stream.read()
    else:
        raise ValueError('Stream must be instance of StringIO or BytesIO, not {}'.format(type(stream)))
    return data


class MeshGeometry(Geometry):
    def __init__(self, contents:str, mesh_format:str):
        super().__init__()
        self.contents = contents
        self.mesh_format = mesh_format

    def lower(self, object_data):
        return {
            "type": "_meshfile_geometry",
            "uuid": self.uuid,
            "format": self.mesh_format,
            "data": self.contents
        }


class ObjMeshGeometry(MeshGeometry):
    def __init__(self, contents:str):
        super().__init__(contents, "obj")

    @staticmethod
    def from_file(fname):
        with open(fname, "r") as f:
            return MeshGeometry(f.read(), "obj")

    @staticmethod
    def from_stream(f):
        return MeshGeometry(data_from_stream(f), "obj")


class DaeMeshGeometry(MeshGeometry):
    def __init__(self, contents:str):
        super().__init__(contents, "dae")  

    @staticmethod
    def from_file(fname):
        with open(fname, "r") as f:
            return MeshGeometry(f.read(), "dae")

    @staticmethod
    def from_stream(f):
        return MeshGeometry(data_from_stream(f), "dae")


class StlMeshGeometry(MeshGeometry):
    def __init__(self, contents:str):
        super().__init__(contents, "stl")

    @staticmethod
    def from_file(fname):
        with open(fname, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            _, extcode = threejs_type(np.uint8)
            encoded = umsgpack.Ext(extcode, arr.tobytes())
            return MeshGeometry(encoded, "stl")

    @staticmethod
    def from_stream(f):
        if isinstance(f, BytesIO):
            arr  = np.frombuffer(f.read(), dtype=np.uint8)
        elif isinstance(f, StringIO):
            arr = np.frombuffer(bytes(f.read(), "utf-8"), dtype=np.uint8)
        else:
            raise ValueError('Stream must be instance of StringIO or BytesIO, not {}'.format(type(f)))
        _, extcode = threejs_type(np.uint8)
        encoded = umsgpack.Ext(extcode, arr.tobytes())
        return MeshGeometry(encoded, "stl")


class TriangularMeshGeometry(Geometry):
    """
    A mesh consisting of an arbitrary collection of triangular faces. To
    construct one, you need to pass in a collection of vertices as an Nx3 array
    and a collection of faces as an Mx3 array. Each element of `faces` should
    be a collection of 3 indices into the `vertices` array.

    For example, to create a square made out of two adjacent triangles, we
    could do:

    vertices = np.array([
        [0, 0, 0],  # the first vertex is at [0, 0, 0]
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],  # The first face consists of vertices 0, 1, and 2
        [3, 0, 2]
    ])

    mesh = TriangularMeshGeometry(vertices, faces)

    To set the color of the mesh by vertex, pass an Nx3 array containing the
    RGB values (in range [0,1]) of the vertices to the optional `color`
    argument, and set `vertexColors=True` in the Material.
    """
    __slots__ = ["vertices", "faces"]

    def __init__(self, vertices:np.ndarray, faces:np.ndarray, color:np.ndarray|None=None):
        super().__init__()

        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        assert vertices.shape[1] == 3, "`vertices` must be an Nx3 array"
        assert faces.shape[1] == 3, "`faces` must be an Mx3 array"
        self.vertices = vertices
        self.faces = faces
        if color is not None:
            color = np.asarray(color, dtype=np.float32)
            assert np.array_equal(vertices.shape, color.shape), "`color` must be the same shape as vertices"
        self.color = color

    def lower(self, object_data):
        attrs = {"position": pack_numpy_array(self.vertices.T)}
        if self.color is not None:
            attrs["color"] = pack_numpy_array(self.color.T)
        return {
            "uuid": self.uuid,
            "type": "BufferGeometry",
            "data": {
                "attributes": attrs,
                "index": pack_numpy_array(self.faces.T)
            }
        }


class PointsGeometry(Geometry):
    def __init__(self, position:np.ndarray, color:np.ndarray|None=None):
        super().__init__()
        self.position = position
        self.color = color

    def lower(self, object_data):
        attrs = {"position": pack_numpy_array(self.position)}
        if self.color is not None:
            attrs["color"] = pack_numpy_array(self.color)
        return {
            "uuid": self.uuid,
            "type": "BufferGeometry",
            "data": {
                "attributes": attrs
            }
        }


class PointsMaterial(Material):
    def __init__(self, size: float = 0.001, color: int = 0xffffff):
        super().__init__()
        self.size = size
        self.color = color

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "PointsMaterial",
            "color": self.color,
            "size": self.size,
            "vertexColors": 2
        }


class Points(Object):
    _type = "Points"


def PointCloud(position: np.ndarray, color: np.ndarray, **kwargs):
    return Points(
        PointsGeometry(position, color),
        PointsMaterial(**kwargs)
    )


def SceneText(text, width=10, height=10, **kwargs):
    return Mesh(
        Plane(width=width,height=height),
        MeshPhongMaterial(map=TextTexture(text,**kwargs),transparent=True,
            needsUpdate=True)
        )

class Line(Object):
    _type = "Line"


class LineSegments(Object):
    _type = "LineSegments"


class LineLoop(Object):
    _type = "LineLoop"


def triad(scale=1.0):
    """
    A visual representation of the origin of a coordinate system, drawn as three
    lines in red, green, and blue along the x, y, and z axes. The `scale` parameter
    controls the length of the three lines.

    Returns an `Object` which can be passed to `set_object()`
    """
    return LineSegments(
        PointsGeometry(position=np.array([
            [0, 0, 0], [scale, 0, 0],
            [0, 0, 0], [0, scale, 0],
            [0, 0, 0], [0, 0, scale]]).astype(np.float32).T,
            color=np.array([
            [1, 0, 0], [1, 0.6, 0],
            [0, 1, 0], [0.6, 1, 0],
            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        ),
        LineBasicMaterial(vertexColors=True))
