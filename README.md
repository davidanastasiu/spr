## SPR: Selective Partitioned Regression

We present our Selective Partitioned Regression (SPR) machine learning model, which was used in our paper `Selective Partitioned Regression for Accurate Kidney Health Monitoring` to efficiently and effectively predict the severity of kidney disease based on the colorimetric change in test strips.

If you make use of our code or data, please cite our paper.

```bibtex
@article{WhelanEBA2024,
    author      = {Alex Whelan and Ragwa Elsayed and Alessandro Bellofiore and David C. Anastasiu},
    title       = {Selective Partitioned Regression for Accurate Kidney Health Monitoring},
    journal     = {Annals of Biomedical Engineering},
    volume      = {},
    year        = {2024},
    number      = {},
    url         = {},
    issn        = {},
    doi         = {10.1007/s10439-024-03470-}
}
```

## How it works

SPR is a specialized regression model for domains with a large range in regression values that are not uniformly distributed across that range. We use SPR for kidney health monitoring but it can be used in any other domain with similar constraints. To achieve superior performance, the SPR model architecture is segmented into two phases. SPR uses a composition of specialized state-of-the-art algorithms as sub-estimators that perform classification and regression. The first phase of our model is a classification task that involves selecting a particular localized regressor that is trained on a subset of partitioned samples. The second phase is a regression task where we use the predicted regressor to estimate the final regression value.


## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.11. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create --name spr python=3.11
conda activate spr
python -m pip install -r env/requirements.txt
```

## Dataset

The dataset is included in the `dataset` directory. Each image is the extracted detection zone from a chemical experiment. Metadata for each sample includes the amount of time since the experiment when the image was taken, the ground truth creatinine concentration, and the biographical data assigned to the sample. 

## Training

First, choose your execution options. Options can be set through a YAML file such as the given `settings.yml`, though the settings object itself (it comes with defaults and works like a dictionary), or through the command line (see example in `scripts/run-spr.py`). The available options are defined in `utils/options.py`.

```python
from utils.options import Options
options = Options(data_dir='dataset', settings='settings.yml', parse_cmdline=False)
```

SPR comes with a data loader that will load and track both the detection zone images and their associated metadata which are needed in eGFR calculations. An options object is needed to initiate the data loader. Note that the data loader will automatically split the dataset into training, validation, and test using the options you specified in the options object.

```python
from utils.data_loader import DataLoader
dl = DataLoader(options)
```

Finally, training can be executed by simply calling the `fit()` method on an SPR object. The SPR object must be initialized with the choice of the estimator that should be used for the underlying classification and regression models. Available estimators include XGBoost, Histogram_Boosted_Trees (or HBT), Random_Forest (or RF), Decision_Tree (or DT), KNN, Logistic_Regression (or LR), and SVM.

```python
from spr import SPR
model = SPR('HBT', dl)
model.fit()
```

## Inference

One can perform inference by calling the model's predict function. Note that the data to be predicted must be available in the data loader, i.e., we must have access to both the images and their associated metadata. If `x` is not defined, `predict()` defaults to the data loader test set. The result of predict are the classification (status) predictions and the regression concentration predictions for each sample in the test set.

```python
status_predictions, concentration_predictions = model.predict(x=dl.X_test)
```

Alternatively, one can simply call the `eval()` method of the object, which will evaluate the performance of the model on the test set, returning both a set of computed evaluation metrics and a data frame with the metadata and predicted values for all test samples.

```python
df, metrics = model.eval()
```

See `example.ipynb` for an example execution of these methods. Alternatively, execute `python scripts/run-spr.py` and provide any optional parameters from the command line.