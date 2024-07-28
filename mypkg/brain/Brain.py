import numpy as np
import os

from . import permute as perm
from . import path as pth


class Brain:
    """A class containing data that represents a single brain.

    Attributes:
        connectome (array): Array of the connectome.
        reducedConnectome (array): Connectome with extreme components culled (?? Add doc please).
        distance_matrix (array): Matrix of distances between brain regions.
        permutation (array): The permutation applied to the data file connectome.
        ordering (type): Description of parameter `ordering`.
        ntf_params (dict): Parameters for the network transfer model.

    """

    def __init__(self):
        # Body variables
        self.connectome = None
        self.reducedConnectome = None
        self.distance_matrix = None
        self.permutation = None
        self.ordering = None
        self.laplacian = None
        self.eigenvalues = None
        self.norm_eigenmodes = None
        self.regular_eigenvalues = None
        self.regular_laplacian = None
        self.norm_regular_eigenmodes = None
        self.raw_regular_eigenvectors = None

        self.ntf_params = {
            "tau_e": 0.012,
            "tau_i": 0.003,
            "alpha": 1.0,
            "speed": 5.0,
            "gei": 4.0,
            "gii": 1.0,
            "tauC": 0.006
#             "beta": 1.0,
        }

    def add_ordering(self, filename):
        standard_list = pth.read_hdf5(filename)
        self.ordering = standard_list

    def add_connectome(
        self,
        hcp_dir,
        conmat_in="mean80_fibercount.csv",
        dmat_in="mean80_fiberlength.csv",
    ):

        self.connectome = np.genfromtxt(
            os.path.join(hcp_dir, conmat_in), delimiter=",", skip_header=1
        )

        self.distance_matrix = np.genfromtxt(
            os.path.join(hcp_dir, dmat_in), delimiter=",", skip_header=0
        )

    def add_ordered_connectome(self, confile, distfile):
        """add a connectome and distance matrix using ordering directly"""
        con, dist, permutation = perm.reorder_connectome(
            conmatfile=confile, distmatfile=distfile
        )
        self.connectome = con
        self.distance_matrix = dist
        self.permutation = permutation

    def reorder_connectome(self, connectome, distancematrix):
        """re-order the present connectome and distance matrix -- note that this is
        a first iteration and some work needs to be done to make it flexible with regards
        the specific ordering."""
        con, dist, permutation = perm.reorder_connectome(
            conmat=connectome, distmat=distancematrix
        )
        self.connectome = con
        self.distance_matrix = dist
        self.permutation = permutation

    def bi_symmetric_c(self):
        """Short summary.

        Args:
            linds (type): Description of parameter `linds`.
            rinds (type): Description of parameter `rinds`.

        Returns:
            type: Description of returned object.

        """
        if self.connectome.shape[0] == 68:
            # Some other ordering that was in the original code:
            linds = np.arange(0, 34)
            rinds = np.arange(34, 68)
        else:
            # Some other ordering that was in the original code:
            linds = np.concatenate([np.arange(0, 34), np.arange(68, 77)])
            rinds = np.concatenate([np.arange(34, 68), np.arange(77, 86)])

        q = np.maximum(
            self.connectome[linds, :][:, linds], self.connectome[rinds, :][:, rinds]
        )
        q1 = np.maximum(
            self.connectome[linds, :][:, rinds], self.connectome[rinds, :][:, linds]
        )
        self.connectome[np.ix_(linds, linds)] = q
        self.connectome[np.ix_(rinds, rinds)] = q
        self.connectome[np.ix_(linds, rinds)] = q1
        self.connectome[np.ix_(rinds, linds)] = q1

    def reduce_extreme_dir(self, max_dir=0.95, f=7):
        """Short summary.

        Args:
            max_dir (type): Description of parameter `max_dir`.
            f (type): Description of parameter `f`.

        Returns:
            type: Description of returned object.

        """
        thr = f * np.mean(self.connectome[self.connectome > 0])
        C = np.minimum(self.connectome, thr)
        C = max_dir * C + (1 - max_dir) * C
        self.reducedConnectome = C
