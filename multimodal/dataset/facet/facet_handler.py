
class FacetHandler(object):
    def __init__(self, facetgroup):
        self.facetgroup = facetgroup

    def is_default(self):
        if 'DefaultFacet' in self.facetgroup:
            return self.facetgroup['DefaultFacet']
        else:
            return False

    def get_samplerate(self):
        raise NotImplementedError()

    def group_name(self):
        return self.facetgroup.name
