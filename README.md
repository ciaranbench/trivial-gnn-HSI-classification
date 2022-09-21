## Trivial hyperspectral image classification with a graph neural network
###Ciaran Bench

There is a strong desire among spectroscopists to exploit the chemical/compositional information encoded in a tissue's hyperspectral image (HSI) (an image that can be formed by collating Raman or IR spectra acquired at equally spaced-out points across a sample) to predict its pathology state well before more obvious morphological signs used by pathologists become visible at later stages of the disease's progression.

Though, the high dimensionality of HSIs demand models with a large number of parameters to manipulate whole images. These require extensive training sets to optimise, which are time-consuming and expensive to acquire. Often patch-based strategies are implemented to make training more computationally tractable. However, these often fail to take into account long-range spatio-spectral features that may be critical to identifying the sample's pathology state.

We discuss the possibility of using a CAE-based feature extraction framework to convert whole HSIs into graph structures, making them amenable to processing with lightweight graph neural networks. This scheme enables the detection of long-range spatial dependencies using fewer parameters than generic convolutional networks applied to whole HSIs. This may ultimately reduce the amount of labelled training data required to learn an accurate classification model.

We test our approach on a trivial toy problem, mostly to demonstrate how this strategy can be implemented as opposed to exhibiting a scenario where it may outperform more generic architectures.
