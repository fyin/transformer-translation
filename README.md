# transformer-translation
A practice to use transformer for language translation.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n transformer-translation python=3.12`
* Activate the environment `conda activate transformer-translation`
* Install the dependencies `conda install --yes --file requirements.txt`

## Modules
* `model`: contains the transformer model implementation code.
* `dataset`: contains the dataset implementation to prepare bilingual data used in sequence to sequence model training and validation.
* `config`: contains the configuration for the model and the training process.
* `train`: contains the code to train and validate the model.

## References
* https://www.youtube.com/watch?v=bCz4OMemCcA 
* https://github.com/hkproj/transformer-from-scratch-notes
* https://github.com/hkproj/pytorch-transformer

