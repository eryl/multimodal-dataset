import h5py
from multimodal.dataset.facet import make_facet, is_facet

class MultiModalDatasets(object):
    """
    Base class for dealing with multi-modal datasets. The datasets are contained in HDF5 files with a particular structure.
    """
    def __init__(self, hdf5_paths):
        self.hdf5_paths = hdf5_paths
        self.datasets = [MultiModalDataset(hdf5_path) for hdf5_path in hdf5_paths]


class Modality(object):
    def __init__(self, name, group):
        self.name = name
        self.group = group
        self.facets = dict()
        self.default_facet = None
        self.setup_facets()

    def setup_facets(self):
        try:
            default_facet_key = self.group.attrs['DefaultFacet']
        except KeyError:
            self.default_facet = None
            default_facet_key = None
        for name, group in self.group.items():
            if is_facet(group):
                facet = make_facet(group)
                if name == default_facet_key or self.default_facet is None:
                    self.default_facet = facet
                self.facets[name] = facet

    def get_facets(self):
        raise NotImplementedError()

    def get_facet(self, id=None):
        return self.facets.get(id, self.default_facet)

    def get_samplerate(self, id=None):
        facet = self.facets.get(id, self.default_facet)
        return facet.get_samplerate()


class MultiModalDataset(object):
    def __init__(self, hdf5_path, mode='r'):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.store = h5py.File(hdf5_path, mode=mode)
        self.modalities = dict()
        self.setup_modalities()

    def setup_modalities(self):
        for name, group in self.store.items():
            modality = Modality(name, group)
            self.modalities[name] = modality

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store.close()

    def close(self):
        self.store.close()

    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield self[i]



