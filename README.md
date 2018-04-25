# PGM-Project
Human Body Pose generation using Factor Graphs

Running the Code
• To change the number of iteration over which we run the sampling over, change line number 7 in
perform_inf erence.py
• To run the training and inference for a Gaussian distribution run the command python main_f n_pose_only.py
• To run the training algorithm for a mixture of Gaussians, run the command; python perf orm_training_mix_gaussian
• To change the number of Gaussian Mixture components, change Line 5 in the file f ind_unary_potentials.py
and find_pairwise_potentials.py
• To perform the inference for a mixture of Gaussians, run the command python perf orm_inf erence_mix_gaussians.py

For SMPL code and dataset
https://ps.is.tuebingen.mpg.de/publications/smpl-2015

Requirements:
mayavi, chumpy, smpl


