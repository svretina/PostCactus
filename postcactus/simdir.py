#!/usr/bin/env python3
""" This module provides easy access to CACTUS data files.

A simulation directory is represented by an instance of the
:py:class:`~.SimDir` class, which provides access to all supported
data types.
"""

import os
# We ideally would like to use cached_property, but it is in Python 3.8
# which is quite new
from functools import lru_cache
from postcactus import cactus_scalars

class SimDir:
    """This class represents a CACTUS simulation directory.

    Data is searched recursively in all subfolders. No particular folder
    structure (e.g. SimFactory style) is assumed. The following attributes
    allow access to the supported data types:

    :ivar path:           Top level path of simulation directory.
    :ivar dirs:           All directories in which data is searched.
    :ivar parfiles:       The locations of all parameter files.
    :ivar initial_params: Simulation parameters, see
                          :py:class:`~.Parfile`.
    :ivar logfiles:       The locations of all log files (.out).
    :ivar errfiles:       The location of all error log files (.err).
    :ivar ts:             Scalar data of various type, see
                          :py:class:`~.ScalarsDir`
    :ivar grid:           Access to grid function data, see
                          :py:class:`~.GridOmniDir`.
    :ivar gwmoncrief:     GW signal obtained using Moncrief formalism,
                          see :py:class:`~.CactusGWMoncrief`.
    :ivar gwpsi4mp:       GW signal from the Weyl scalar multipole
                          decomposition, see :py:class:`~.CactusGWPsi4MP`.
    :ivar emphi2mp:       EM signal from the Weyl scalar multipole
                          decomposition, see :py:class:`~.CactusEMPhi2MP`.
    :ivar ahoriz:         Apparent horizon information, see
                          :py:class:`~.CactusAH`.
    :ivar multipoles:     Multipole components, see
                          :py:class:`~.CactusMultipoleDir`.
    :ivar metadata:       This allows augmenting the simulation folder
                          with metadata, see :py:class:`~.MetaDataFolder`.
    :ivar timertree:      Access TimerTree data, see
                          :py:class:`~.TimerTree`.
    """

    def _sanitize_path(self, path):
        # Make sure to have complete paths with respect to the current folder
        self.path = os.path.abspath(os.path.expanduser(path))
        if (not os.path.isdir(self.path)):
            raise RuntimeError(f"Folder does not exist: {path}")

    def _scan_folders(self, max_depth):
        """Scan all the folders in self.path up to depth max_depth
        and categorize all the files.
        """

        self.dirs = []
        self.parfiles = []
        self.logfiles = []
        self.errfiles = []
        self.allfiles = []

        def listdir_no_symlinks(path):
            """Return a list of files in path that are not symlink

            """
            dir_content = [os.path.join(path, p) for p in os.listdir(path)]
            return [p for p in dir_content if not os.path.islink(p)]

        def filter_ext(files, ext):
            """Return a list from the input list of file that
            has file extension ext."""
            return [f for f in files if os.path.splitext(f)[1] == ext]

        def walk_rec(path, level=0):
            """Walk_rec is a recursive function that steps down all the
            subdirectories (except the ones with name defined in self.ignore)
            up to max_depth and add to self.allfiles the files found in the
            directories.

            """
            if (level >= max_depth):
                return

            self.dirs.append(path)

            all_files_in_path = listdir_no_symlinks(path)

            files_in_path = list(filter(os.path.isfile, all_files_in_path))
            self.allfiles += files_in_path

            directories_in_path = list(filter(os.path.isdir,
                                              all_files_in_path))

            # We ignore the ones in self.ignore
            directories_to_scan = [p for p in directories_in_path if
                                   (os.path.basename(p) not in self.ignore)]

            # Apply walk_rec to all the subdirectory, but with level increased
            for p in directories_to_scan:
                walk_rec(p, level + 1)

        walk_rec(self.path)

        self.logfiles = filter_ext(self.allfiles, '.out')
        self.errfiles = filter_ext(self.allfiles, '.err')
        self.parfiles = filter_ext(self.allfiles, '.par')

        # Sort by time
        self.parfiles.sort(key=os.path.getmtime)
        self.logfiles.sort(key=os.path.getmtime)
        self.errfiles.sort(key=os.path.getmtime)

        simfac = os.path.join(self.path, 'SIMFACTORY', 'par')

        # Simfactory has a folder SIMFATORY with a subdirectory for par files
        # Even if SIMFACTORY is excluded, we should include that par file
        if os.path.isdir(simfac):
            mainpar = filter_ext(listdir_no_symlinks(simfac), '.par')
            self.parfiles = mainpar + self.parfiles

        self.has_parfile = bool(self.parfiles)

        # TODO: Add this when cactus_parfile is ready

        # if self.has_parfile:
        #     self.initial_params = cpar.load_parfile(self.parfiles[0])
        # else:
        #     self.initial_params = cpar.Parfile()

    def __init__(self, path, max_depth=8, ignore=None):
        """Constructor.

        :param path:      Path to simulation directory.
        :type path:       string
        :param max_depth: Maximum recursion depth for subfolders.
        :type max_depth:  int
        :param ignore: Folders to ignore
        :type ignore:  set

        Parfiles (*.par) will be searched in all data directories and the
        top-level SIMFACTORY/par folder, if it exists. The parfile in the
        latter folder, if available, or else the oldest parfile in any of
        the data directories, will be used to extract the simulation
        parameters. Logfiles (*.out) and errorfiles (*.err) will be
        searched for in all data directories.
        """
        if (ignore is None):
            ignore = {'SIMFACTORY', 'report', 'movies', 'tmp', 'temp'}

        self.ignore = ignore
        self._sanitize_path(str(path))
        self._scan_folders(int(max_depth))

    @property
    # We only need to keep it 1 in memory: it is the only possible!
    @lru_cache(1)
    def ts(self):
        return cactus_scalars.ScalarsDir(self)

    timeseries = ts

    def __str__(self):
        header = f"Indexed {len(self.allfiles)} files"
        header += f"and {len(self.dirs)} subdirectories\n"

        ts_ret = self.ts.__str__()

        return header + ts_ret