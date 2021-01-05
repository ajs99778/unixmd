from __future__ import division
from qm.qm_calculator import QM_calculator
from misc import call_name

class QChem(QM_calculator):
    """ Class for common parts of QChem program

        :param string basis_set: basis set information
        :param string memory: allocatable memory in the calculations
        :param string qm_path: path for QChem
        :param integer nthreads: number of threads in the calculations
        :param string version: version of QChem program
    """
    def __init__(self, basis_set, memory, qm_path, nthreads, version):
        # Save name of QM calculator and its method
        super().__init__()

        # Initialize QChem common variables
        self.basis_set = basis_set

        self.memory = memory
        self.qm_path = qm_path
        self.nthreads = nthreads
        self.version = version

        if (self.version != "5.2"):
            raise ValueError (f"( {self.qm_prog}.{call_name()} ) Other version not implemented! {self.version}")
