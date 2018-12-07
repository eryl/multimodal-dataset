import h5py
from multimodal.dataset.facet import make_facet, is_facet

class MultiModalDatasets(object):
    """
    Base class for dealing with multi-modal datasets. The datasets are contained in HDF5 files with a particular structure.
    """
    def __init__(self, hdf5_paths):
        self.hdf5_paths = hdf5_paths


class Modality(object):
    def __init__(self, group):
        self.group = group
        self.facets = []
        self.setup_facets()

    def setup_facets(self):
        for name, group in self.group.items():
            if is_facet(group):
                facet = make_facet(group)
                self.facets.append(facet)

    def get_facets(self):
        raise NotImplementedError()


class MultiModalDataset(object):
    def __init__(self, hdf5_path, mode='r'):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.store = h5py.File(hdf5_path, mode=mode)
        self.modalities = []
        self.setup_modalities()

    def setup_modalities(self):
        for name, group in self.store.items():
            modality = Modality(group)
            self.modalities.append(modality)



