# mlr-xai-selfies
SELFIES mutation method to obtain atom attributions for any QSAR model.

### What are SELFIES?
SELFIES (SELF-referencing Embedded Strings) are a string representation for molecules. More information can be found in the [paper](https://arxiv.org/abs/1905.13741) and the code to compute SELFIES from molecules is available in [github](https://github.com/aspuru-guzik-group/selfies).
SELFIES differ from SMILES in the fact that they are backed by a grammar which ensures chemical validity. Meaning that any valid SELFIES string is also a valid molecule.

### What is the idea behind xai-selfies?
XAI-SELFIES can be viewed as a generalization of the XAI method published in [this paper](https://www.sciencedirect.com/science/article/pii/S2667318522000174) and can be considered an outcome of the Bayer LSC project "Explainable AI". 

The general concept is to explain any trained QSAR model using string permutations to obtain character-level attribution scores.
The overall algorithm is the following

```For each input molecule of interest:
        Obtain the corresponding SELFIES string
        Obtain the prediction from the model to explain
        For each position in the string:
             Mutate the string at the position of interest by replacing the SELFIES character by all possible 
             characters in the SELFIES vocabulary
             Check for SELFIES validity
             Optionally check for distance to input molecule
             Obtain predictions for all valid mutated strings
             Attribution_for_position_i = original prediction - average(mutated predictions)
        convert the SELFIES attributions into atom attributions by using SELFIES-to-SMILES correspondences
 ```
          
### How do I get started?

- Create a conda environment with all necessary dependencies using the environment.yml file:
```conda env create -f environment.yml```

- Have a look at example.py: by running it you will download a public logD dataset, create a demo QSAR model based on this dataset, and create attribution vectors for the first 200 molecules of the dataset. It shows how the pretrained model should look like and how the featurizer should look like. 

### How can I visualize attributions computed with XAI-SELFIES?

Several ways! 

- The first one would be to use the RDKit library, specifically by using the SimilarityMaps functionality as shown [here](https://www.rdkit.org/docs/GettingStartedInPython.html#generating-similarity-maps-using-fingerprints).

- Another option is to use the beautiful [xSMILES library](https://github.com/Bayer-Group/xsmiles) published by Henry Heberle, which can work as a jupyter notebook plugin.

- Finally we have also built [CIME](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00600-z), a visual analytics platform which integrates xSMILES in a webapp, and lets you upload datasets as csv format (i.e., you can just save the pandas dataframe obtained from running XAI-SELFIES as an sdf and move on to CIME to analyze your data. The public version of CIME is available [here](https://github.com/jku-vds-lab/cime) and can be launched as a docker container.

### Acknowledgements
Code developed by Floriane Montanari while employed in the Machine Learning Group at Bayer.
Kudos to Linlin Zhao (whose [xBCF implementation](https://github.com/Bayer-Group/xBCF) helped make XAI-SELFIES), Marco Bertolini and Thomas Wolf for contibuting ideas!

