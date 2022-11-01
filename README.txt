CT-ORG: CT volumes with multiple organ segmentations

DESCRIPTION

This dataset consists of 140 computed tomography (CT) scans, each with five organs labeled in 3D: lung, bones, liver, kidneys and bladder. The brain is also labeled on the minority of scans which show it.

Patients were included based on the presence of lesions in one or more of the labeled organs. Most of the images exhibit liver lesions, both benign and malignant. Some also exhibit metastatic disease in other organs such as bones and lungs.

The images come from a wide variety of sources, including abdominal and full-body; contrast and non-contrast; low-dose and high-dose CT scans. 131 images are dedicated CTs, the remaining 9 are the CT component taken from PET-CT exams. This makes the dataset ideal for training and evaluating organ segmentation algorithms, which ought to perform well in a wide variety of imaging conditions.

The dataset includes large and easily-located organs such as the lungs, as well as small and difficult ones like the bladder. We hope the dataset will enable widespread adoption of multi-class organ segmentation, as well as competitive benchmarking of algorithms for it.

The data are divided into a testing set of 21 CT scans, and a training set of the remaining 119. For the training set, the lungs and bones were automatically segmented by morphological image processing. For the testing set, the lungs and bones were segmented manually. All other organs were segmented manually in both the training and testing sets. Manual segmentations were done with ITK-SNAP (https://www.itksnap.org), starting with semi-automatic active contour segmentation followed by manual clean-up. The source code for the morphological algorithms is available at:
        - https://github.com/bbrister/ctOrganSegmentation.git

Many images were borrowed from the Liver Tumor Segmentation (LiTS) challenge, which the organizers have generously allowed us to distribute. For more information, see the following website and paper:
        - https://lits-challenge.com
        - Arxiv [1901.04056] The Liver Tumor Segmentation Benchmark (LiTS) (https://arxiv.org/abs/1901.04056)

DATA FORMAT

All files are stored in Nifti-1 format with 32-bit floating point data. 

Images are stored as 'volume-XX.nii.gz' where XX is the case number. All images are CT scans, under a wide variety of imaging conditions including high-dose and low-dose, with and without contrast, abdominal, neck-to-pelvis and whole-body. Many patients exhibit cancer lesions, especially in the liver, but they were not selected according to any specific disease criteria. Numeric values are in Hounsfield units.

Segmentations are stored as 'labels-XX.nii.gz', where XX is the same number as the corresponding volume file. Organs are encoded as follows:

0: Background (None of the following organs)
1: Liver
2: Bladder
3: Lungs
4: Kidneys
5: Bone
6: Brain

TEST AND TRAIN SPLITS

All organ masks were generated either (A) semi-automatically using ITK-SNAP, or (B) automatically using morphological algorithms. ITK-SNAP is a popular open-source program for medical image segmenation. Semi-automatic segmentation consists of manual editing with the 3D paintbrush tool, followed by refinement with active contours.

The first 21 volumes (case numbers 0-20) constitute the TESTING split. All organs in these volumes have been labeled with method (B). Bones were first labeled with method (A), then the result was refined with method (B).

The remaining volumes constitute the TRAINING split. For these volumes, both lungs and bones were labeled with method (B). These masks suffice for training a deep neural network, but should not be considered reliable for evaluation.

All other organs were labeled with method (A) for both the training and testing splits. For these organs, there is no difference in label accuracy between the two splits. 

SOFTWARE

Code used to generate the unsupervised lung and bone segmentations can be found at:
        - https://github.com/bbrister/ctOrganSegmentation.git

CREDITS

These data were annotated between 2018-2019 by:
        -Blaine Rister
        -Kaushik Shivakumar

131 of the original images came from the Liver Tumor Segmentation Challenge (LiTS). Please see the challenge website (https://competitions.codalab.org/competitions/17094) for the credits for these images. Most of the liver masks for these images also came from LiTS, although some were annotated by the above. 9 additional images were added from PET-CT patients from Stanford Healthcare, so that this additional imaging modality could be represented in the training and evaluation data.

Manual annotations were made using ITK-SNAP. It is available at https://www.itksnap.org.

Please direct questions to Blaine Rister by email at blaine@stanford.edu.

CITATIONS

Please refer to the following paper to cite this data:
        - Arxiv [1901.04056] The Liver Tumor Segmentation Benchmark (LiTS)
(https://arxiv.org/abs/1901.04056)

BIBTEX

The previous article in Bibtex, for your convenience:

@article{DBLP:journals/corr/abs-1901-04056,
  author    = {Patrick Bilic and
               Patrick Ferdinand Christ and
               Eugene Vorontsov and
               Grzegorz Chlebus and
               Hao Chen and
               Qi Dou and
               Chi{-}Wing Fu and
               Xiao Han and
               Pheng{-}Ann Heng and
               J{\"{u}}rgen Hesser and
               Samuel Kadoury and
               Tomasz K. Konopczynski and
               Miao Le and
               Chunming Li and
               Xiaomeng Li and
               Jana Lipkov{\'{a}} and
               John S. Lowengrub and
               Hans Meine and
               Jan Hendrik Moltz and
               Chris Pal and
               Marie Piraud and
               Xiaojuan Qi and
               Jin Qi and
               Markus Rempfler and
               Karsten Roth and
               Andrea Schenk and
               Anjany Sekuboyina and
               Ping Zhou and
               Christian H{\"{u}}lsemeyer and
               Marcel Beetz and
               Florian Ettlinger and
               Felix Gr{\"{u}}n and
               Georgios Kaissis and
               Fabian Loh{\"{o}}fer and
               Rickmer Braren and
               Julian Holch and
               Felix Hofmann and
               Wieland H. Sommer and
               Volker Heinemann and
               Colin Jacobs and
               Gabriel Efrain Humpire Mamani and
               Bram van Ginneken and
               Gabriel Chartrand and
               An Tang and
               Michal Drozdzal and
               Avi Ben{-}Cohen and
               Eyal Klang and
               Michal Marianne Amitai and
               Eli Konen and
               Hayit Greenspan and
               Johan Moreau and
               Alexandre Hostettler and
               Luc Soler and
               Refael Vivanti and
               Adi Szeskin and
               Naama Lev{-}Cohain and
               Jacob Sosna and
               Leo Joskowicz and
               Bjoern H. Menze},
  title     = {The Liver Tumor Segmentation Benchmark (LiTS)},
  journal   = {CoRR},
  volume    = {abs/1901.04056},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.04056},
  archivePrefix = {arXiv},
  eprint    = {1901.04056},
  timestamp = {Fri, 03 May 2019 12:59:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-04056},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
