# Multimodal dataset
The multimodal datasets are contained in hdf5 files.

The datasets contain one hdf5 group per modality. Each modality can in turn have different facets which are also hdf5 groups. Each facet is the modality expressed in different ways. For example, a video modality could have two facets with different frame rates, different sizes or different camera angles. A subtitle modality could have different facets 
for different languages and a sound modality could have different facets for different audio tracks.
Each facet can contain multiple hdf5 data arrays. How to interpret them is modality-specific. A subtitle facet for example 
contains one data array with the text fragments, one with timestamps for the fragments and one with metadata for the fragments. 
Each facet has an attrs dictionary which must contain these fields:

* 'FacetHandler': Name of the python class which handles this kind of facet.
* 'HandlerVersion': The version of the handler class. This field can give the user a hint of 
                    what class to use for loading the data properly.

* Modality 1
  * Facet 1
    * Data array 1
    * Data array 2
  * Facet 2
    * Data array 1
    * Data array 2
    * Data array 3
  * Channel n