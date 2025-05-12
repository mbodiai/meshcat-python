import pathlib
from urllib.parse import urlparse
try:
    from pydantic import AnyUrl
except ImportError:
    AnyUrl = str

class Path(object):
    __slots__ = ("entries","scheme","netloc","params","query","fragment")

    def __init__(self, entries=tuple(), scheme:str|None=None):
        self.entries = entries
        self.scheme = scheme

    def append(self, other):
        new_path = self.entries
        for element in other.split('/'):
            if len(element) == 0:
                new_path = tuple()
            else:
                new_path = new_path + (element,)
        return Path(new_path)

    def lower(self):
        return "/" + "/".join(self.entries)

    def __hash__(self):
        return hash(self.entries)

    def __eq__(self, other):
        return self.entries == other.entries

    def __truediv__(self, other):
        return self.append(other)

    def __rtruediv__(self, other):
        return other.append(self)

    def __div__(self, other):
        return self.append(other)

    def __rdiv__(self, other):
        if not isinstance(other, (Path, pathlib.Path, AnyUrl, str)):
            raise ValueError("Cannot append to non-Path object")
        if isinstance(other, Path):
            self.entries = other.entries + self.entries
        elif isinstance(other, pathlib.Path):
            self.entries = tuple(other.parts) + self.entries
        elif isinstance(other, AnyUrl|str):
            self.entries = (urlparse(str(other)).path,) + self.entries
            self.scheme = urlparse(str(other)).scheme
            self.netloc = urlparse(str(other)).netloc
            self.params = urlparse(str(other)).params
            self.query = urlparse(str(other)).query
            self.fragment = urlparse(str(other)).fragment
        else:
            self.entries = (other,) + self.entries
        return self
        
