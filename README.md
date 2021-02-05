# Multimodal Ophthalmoscopic Image Fusion Using Paired Autoencoders

**Background**

A primary cause of vision loss in the aging population, age-related macular degeneration (AMD) is a progressive retinal condition characterized by the presence of drusen, accumulated deposits of extracellular material between the ocular membranes.
Current AMD studies largely rely on fundus photography to identify visible drusen as a late-stage indicator of disease progression.

Fluorescence lifetime imaging ophthalmoscopy (FLIO) has been posited as a novel source of diagnostic and research data. Through recording changes in fundus autofluorescence, FLIO  data indicates the presence of AMD-related biochemical components and processes. FLIO data analysis has the potential to identify early stages of AMD development.

**Goals**

1) Preprocess fundus and FLIO data into appropriate formats and similar sizes
2) Create paired autoencoder model with constrained feature spaces to take data from these two sources; output single image capturing information from both sources
3) Test different model architectures and loss functions to optimize results 

**Methods**

Preprocessing

1) Fundus
2) FLIO

Utilizing a paired dataset comprising both imaging modalities, we generated fundus-like RGB images that qualitatively highlight the fluorescence lifetime component differences found in the matched FLIO data.

1) Traditional autoencoder for image reconstruction
2) Paired autoencoders joined at the feature space with different architectures and loss functions
3) Visualization

We implemented paired autoencoders for fundus images and FLIO triexponential decay parameters; the autoencoders learned via individual L1-regularized MS-SSIM loss functions and a cosine similarity loss linking the feature spaces.

Current results comprise decoded fundus images constrained by FLIO data that fuse information provided by both modalities, but yield suboptimal resolution and interpretation clarity. Loss experimentation with a SSIM-variant loss or perceptual loss instead of cosine similarity would provide basis for future improvement.

With further refinement, this approach could eventually identify ambiguous or early AMD presentation before pathology is clear to the human eye, leading to timely diagnosis and intervention.

**Packages and Technologies**

1) SimpleITK
2) PyTorch
3) NYU High Performance Computing Cluster
