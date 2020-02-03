"""
This module contains the definition of all the data structures used by our model, :py:class:`hepqpr.qallse.qallse.Qallse`
as well as some useful type alias used throughout the project.
"""

import math
from typing import Set, Iterable

import numpy as np

from .type_alias import *
from .utils import curvature, angle_diff
from dimod import *

class Volayer:
    """
    Support the encoding of a hit's `volume_id` and `layer_id` into one single number that can be used for
    ordering and distance calculation. Note that it only works for TrackML data limited to the barrel region.
    """

    #: Define the mapping of `volume_id` and `layer_id` into one number (the index in the list)
    ordering = [(8, 2), (8, 4), (8, 6), (8, 8), (13, 2), (13, 4), (13, 6), (13, 8), (17, 2), (17, 4)]
    
    #: Define slices in eta and phi

    eta_increment = 3
    eta_overlap = 0.3
    eta_slices = []
    for x in range(int(9/eta_increment)):
        print(x)
        eta_slices.append((x*eta_increment - 4.5, (x*eta_increment + eta_increment + eta_overlap - 4.5)))
    eta_slices.append((4.5, float("inf")))
    eta_slices.append((-float("inf"), -4.5 + eta_overlap))
    print(eta_slices)
    #eta_slices = [(-float("inf"), float("inf"))]


    phi_increment = 0.5
    phi_overlap = 0.1
    phi_slices = []
    for x in range(int(2/phi_increment)):
        phi_slices.append((x*phi_increment, x*phi_increment + phi_increment + phi_overlap))
        print(phi_slices)

    #phi_slices = [(0, 0.5), (0.0, 1.0), (1, 1.5), (1.5, 2)]

    @classmethod
    def get_index(cls, volayer: Tuple[int, int]) -> int:
        """Convert a couple `volume_id`, `layer_id` into a number (see :py:attr:`~ordering`)."""
        return cls.ordering.index(tuple(volayer))
    
    @classmethod
    def get_eta_slice(cls, zval: float,  xval: float, yval: float) -> list:

        # broken eta = (-1)*(zval/np.abs(zval))*np.log(np.abs(np.tan(np.sqrt(xval**2+yval**2)/zval * 0.5)))
        eta = (-1) * (zval / np.abs(zval)) * np.log(np.abs(np.sqrt(xval ** 2 + yval ** 2) / zval * 0.5))
        print(xval, yval, eta)
        # Determine the eta slice the hit belongs to
        etaslices = list(filter(lambda sl: eta>sl[0] and eta<=sl[1], cls.eta_slices))

        etaslice_indices = []
        #populate etaslice indices with cls.etaslices
        for slice in etaslices:
            etaslice_indices.append(cls.eta_slices.index(slice))
        """Get eta-slice index for hit (see :py:attr:`~slices`)."""
        return etaslice_indices
    
    @classmethod
    def get_phi_slice(cls, xval: float, yval: float) -> list:
        """Get phi-slice index for hit (see :py:attr:`~slices`)."""
        phi = np.arctan2(yval, xval)/np.pi+1
        # Determine which phi slice the hit belongs to
        phislices = list(filter(lambda sl: phi>sl[0] and phi<=sl[1], cls.phi_slices))

        # if phi belongs to the first phi slice and is within the last phi slice overlap region
        if phi <= cls.phi_slices[0][0] + cls.phi_overlap:

            phislices.append(cls.phi_slices[len(cls.phi_slices) - 1])

        phislice_indices = []
        # populate phislice_indices with each respective cls.phislice index
        for slice in phislices:
            phislice_indices.append(cls.phi_slices.index(slice))


        return phislice_indices

    @classmethod
    def difference(cls, volayer1, volayer2) -> int:
        """Return the distance between two volayers."""
        return cls.ordering.index(volayer2) - cls.ordering.index(volayer1)


class Xplet:
    """
    Base class for doublets, triplets and quadruplets.
    An xplet is an ordered list of hits (ordered by radius).

    It contains lists of inner and outer xplets (with one more hit) and sets of "kept" inner and outer xplets,
    i.e. xplets actually used when generating the qubo. Those lists and sets are populated during model building
    (see :py:meth:`hepqpr.qallse.Qallse.build_model`).
    """

    def __init__(self, hits, inout_cls=None):
        """
        Create an xplet. Preconditions:
        * hits are all different
        * hits are ordered in increasing radius in the X-Y plane
        """
        self.hits = hits

        if inout_cls is not None:
            self.inner: List[inout_cls] = []
            self.outer: List[inout_cls] = []
            self.inner_kept: Set[inout_cls] = set()
            self.outer_kept: Set[inout_cls] = set()

    def hit_ids(self) -> TXplet:
        """Convert this xplet into a list of hit ids."""
        return [h.hit_id for h in self.hits]

    @classmethod
    def name_to_hit_ids(cls, str):
        """Convert a string representation of an xplet into a list of hit ids (see :py:meth:~`__str__`)."""
        return [int(h) for h in str.split('_')]

    @classmethod
    def hit_ids_to_name(cls, hits):
        """Inverse of :py:meth:~`name_to_hit_ids`."""
        return '_'.join(map(str, hits))

    def __str__(self):
        """Return a string made of hit ids joined by an underscore. This can be used in the QUBO as an identifier."""
        return '_'.join(map(str, self.hits))

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        d = dict(name=str(self), hits=self.hit_ids())
        for k, v in self.__dict__.items():
            if k == 'hits' or k.startswith('inner') or k.startswith('outer'): continue
            if isinstance(v, Xplet): v = str(v)
            d[k] = v
        return d



class Hit(Xplet):
    """One hit."""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        super().__init__([self], Doublet)

        #: The hit id
        self.hit_id: int = int(self.hit_id)
        #: The volayer
        self.volayer: int = Volayer.get_index((int(self.volume_id), int(self.layer_id)))

        #: TODO allow xplets to have multiple slice indices for overlapping slices
        #: The slices, list of the overlapped slice indices
        self.phi_slice: int = Volayer.get_phi_slice(self.x, self.y)
        self.eta_slice: int = Volayer.get_eta_slice(self.z, self.x, self.y)
        
        #: The coordinates in the X-Y plane, i.e. `(x,y)`
        self.coord_2d: Tuple[float, float] = np.array([self.x, self.y])
        #: The coordinates, i.e. `(x,y,z)`
        self.coord_3d: Tuple[float, float, float] = np.array([self.x, self.y, self.z])
        
        #FIXME obsolete?
        # TODO: remove if QallseCs is discarded from the project
        # test: second order conflicts
        #self.inner_tplets: List[Triplet] = []
        #self.outer_tplets: List[Triplet] = []

    def __str__(self):
        return str(self.hit_id)  # to avoid recursion


class Doublet(Xplet):
    """A doublet is composed of two hits."""

    def __init__(self, hit_start: Hit, hit_end: Hit):
        """
        Create a doublet.
        """
        assert hit_start != hit_end
        assert hit_start.r <= hit_end.r

        super().__init__([hit_start, hit_end], Triplet)
        #: The hits composing this doublet
        self.h1, self.h2 = self.hits
        #: The delta r of the doublet
        self.dr = hit_end.r - hit_start.r
        #: The delta z of the doublet
        self.dz = hit_end.z - hit_start.z

        #: The slice in which the doublet belongs to
        self.phi_slice = hit_end.phi_slice
        self.eta_slice = hit_end.eta_slice

        #: The angle in the R-Z plane between this doublet and the R axis.
        self.rz_angle = math.atan2(self.dz, self.dr)
        #: The 2D vector of this doublet in the X-Y plane, i.e. `(∆x,∆y)`
        self.coord_2d = hit_end.coord_2d - hit_start.coord_2d
        #: The 3D vector of this doublet, i.e. `(∆x,∆y,∆z)`
        self.coord_3d = hit_end.coord_3d - hit_start.coord_3d


#: Container carrying all qubo slices
class DoubletContainer(object):
    def __init__(self):
        self.DoubletList = []


    def __str__(self):
        return 'Container with {0} slices'.format(len(self.quboList))

    def addQubo(self, hit_start: Hit, hit_end: Hit, hit_eta: int, hit_phi: int):

        self.DoubletList.append(d)


class Triplet(Xplet):
    """A triplet is composed of two doublets, where the first ends at the start of the other."""

    def __init__(self, d1: Doublet, d2: Doublet):
        """
        Create a triplet. Preconditions:
        * `d1` ends where `d2` starts: `d1.hits[-1] == d2.hits[0]`
        """
        super().__init__([d1.h1, d2.h1, d2.h2], Quadruplet)
        assert d1.hits[-1] == d2.hits[0]
        assert d1.h1.r < d2.h2.r

        self.d1: Doublet = d1
        self.d2: Doublet = d2

        #: The slice in which the triplet belongs to
        self.phi_slice = d2.phi_slice
        self.eta_slice = d2.eta_slice

        #: TODO Identifiy differences between implied helix curvature and menger curvature for impact parameter performance
        #: Radius of curvature, see `Menger curvature <https://en.wikipedia.org/wiki/Menger_curvature>`_.
        self.curvature = curvature(*[h.coord_2d for h in self.hits])
        #: Difference between the doublet's rz angles (see :py:attr:~`Doublet.rz_angle`)
        self.drz = angle_diff(d1.rz_angle, d2.rz_angle)
        #: Sign of the `drz` difference
        self.drz_sign = 1 if abs(d1.rz_angle + self.drz - d2.rz_angle) < 1e-3 else -1
        #: QUBO weight, assigned later
        self.weight = .0

    def doublets(self) -> List[Doublet]:
        """Return the ordered list of doublets composing this triplet."""
        return [self.d1, self.d2]


class Quadruplet(Xplet):
    """A quadruplet is composed of two triplets having two hits (or one doublet) in common."""

    def __init__(self, t1: Triplet, t2: Triplet):
        """
        Create a quadruplet. Preconditions:
        * `t1` and `t2` share two hits/one doublet: `t1.hits[-2:] == t2.hits[:2]` and `t1.d2 == t2.d1`
        """
        assert t1.d2 == t2.d1

        super().__init__(t1.hits + [h for h in t2.hits if h not in t1.hits])

        self.t1: Triplet = t1
        self.t2: Triplet = t2

        #: The slice in which the quadruplet belongs to
        self.phi_slice = t2.phi_slice
        self.eta_slice = t2.eta_slice

        #: Absolute difference between the two triplets' curvatures
        self.delta_curvature = abs(self.t1.curvature - self.t2.curvature)
        #: Number of layers this quadruplet spans across
        #: If no layer skip, this is equal to `len(self.hits) - 1`.
        self.volayer_span = self.hits[-1].volayer - self.hits[0].volayer
        #: QUBO coupling strength between the two triplets. Should be negative to encourage
        #: the two triplets to be kept together.
        self.strength = .0

    def doublets(self) -> List[Doublet]:
        """Return the ordered list of doublets composing this triplet."""
        #: Does this need to be reworked with the slicing algorithm?
        return self.t1.doublets() + [self.t2.d2]


class QuboSlice(object):

    def __init__(self, q: {}, eta: int, phi: int):
        self.qubo = q
        self.eta = eta
        self.phi = phi
    
    def __str__(self):
        return 'eta: {0}, phi: {1}\nQubo: {2}'.format(self.eta,self.phi,self.qubo)


#: Container carrying all qubo slices
class SliceContainer(object):
    def __init__(self):
        self.quboList = []
    
    def __str__(self):
        return 'Container with {0} slices'.format(len(self.quboList))
    
    def addQubo(self, q: QuboSlice):
        self.quboList.append(q)
    
    def getFirstNonEmptyQubo(self):
        for qslice in self.quboList:
            if len(qslice.qubo)>0: 
                print("Return qubo with length {0} and eta: {1}, phi: {2}".format(len(qslice.qubo), qslice.eta, qslice.phi))
                return qslice
        print("All quboSlices empty!")

    def getQubo(self, eta: int, phi: int):
        foundQubo = None
        for qslice in self.quboList:
            if qslice.eta == eta and qslice.phi == phi:
                if not foundQubo: foundQubo = qslice.qubo
                else: raise Exception("ERROR: Found two qubos with same slice coords.")
        if bool(foundQubo):
            print("WARNING: Returning EMPTY qubo, code might crash!")
        return foundQubo


class ResponseSlice(object):

    def __init__(self, r: {}, eta: int, phi: int):


            self.respond = r
            self.eta = eta
            self.phi = phi

    '''def __init__(self, r: Dict[Tuple, int], eta: int, phi: int):
        super(ResponseSlice, self).__init__(r: Dict[Tuple, int], eta: int, phi: int)

            self.respond = r
            self.eta = eta
            self.phi = phi
'''


class ResponseContainer(object):

    def __init__(self):
        self.responseList = []

    def addResponse(self, r: ResponseSlice):
        self.responseList.append(r)

    def getResponse(self, eta: int, phi: int):
        foundResponse = None
        for rslice in self.responseList:
            if rslice.eta == eta and rslice.phi == phi:
                if not foundResponse: foundResponse = rslice.respond
                else: raise Exception("ERROR: Found two responses with same slice coords.")
        if bool(foundResponse):
            print("WARNING: Returning EMPTY qubo response, code might crash!")
        return foundResponse

    def getFirstNonEmptyResponse(self):
        for rslice in self.responseList:
            if len(rslice.respond)>0:
                #print("Return response with length {0} and eta: {1}, phi: {2}".format(len(rslice.qubo), rslice.eta, rslice.phi))
                return rslice
        print("All Responses are empty!")



