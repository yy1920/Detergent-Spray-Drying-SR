# Detergent-Spray-Drying-SR
Code to go along with my MISCADA dissertation
GPL2, Syrbo, GPLB2 require installing gplearn. This can be bypassed if the gplearn is copied to their folders as well. To run the run_gplearn files, it is necessary to put the location of the training files in the train_files list which currently stores the output of glob.
The dataset file names should ..._train.txt , ..._test.txt and for extrapolation the first file should be ..._extrap_1.txt and so on.
The run_gplearn.slurm files can be submitted using the sbatch command. It should be noted the default cpus requested is 32 for speed so this may need changing.
The GPL4 algorithm was found to be the best especially when using reduced variable set/dimensional analysis.
