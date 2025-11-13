Hi! i don't know how to write a README stay tuned for when i find out.

This repository contains the scripts and files required to use a novel method for automatic Antarctic fast-ice detection from Sentinel-1 SAR image pairs.

  - Scripts should be run in the Conda environment sam_env, availabel in sam_env.yml
  - SAR EWs can be acessed at /g/data/yp75/projects/sar-antarctica-processing/utas/

NormProd.ipynb 
  - Contains step 1 of the method (image pair processing)
  - Requires the file process_image_pairs.py
  - Runs on a CPU
  - ARE settings for NormProd.ipynb: Queue = normal, compute size = medium, jobfs size = 10GB, conda environment = sam_env


SAM-SVM.ipynb
  - Contains steps 2 & 3 of the method (segmentation using SAM, classificaiton using SVM)
  - Runs on a GPU
  - ARE settings for NormProd.ipynb: Queue = gpuvolta, compute size = 1 gpu, jobfs size = 10GB, conda environment = sam_env
  - requires coastline shapefiles from add_coastline_high_res_polygon_v7_8.shp.zip, available for download at https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=c7fe759d-e042-479a-9ecf-274255b4f0a1
  - requires SAM model weights (download from the SAM repository at https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints because it's too big for me to upload here)
  - requires SVM training dataset which you can get at /g/data/jk72/gb4219/honours_data/SVM_trainingdata (also because I can't figure out how to put it here)
