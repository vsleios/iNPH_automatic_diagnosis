# iNPH_automatic_diagnosis
1) Master thesis project (Uppsala University 2024, MSc Data Science).
2) Scripts for papers to be published.

## What is iNPH?
![healthy vs iNPH](./images/healthy_vs_iNPH.png)

iNPH (idiopathic Normal Pressure Hydrocephalus) is a neurodegenerative disease that leads to dementia. iNPH is, mainly, characterized by the dilation of the Ventricles.

## Aim of our project

The aim is twofold:

1) Develop Machine Learning models that predict the probability of a patient having iNPH.
2) Investigate the patterns on the brain that characterize iNPH.

## A few results of our study

### Volmetric approach :
![ftt folds](./images/FTTncv.png)

We used a *FeautreTokenizer-Transformer* model, modified for tabular data. We achieved an accuracy of **94.1 %** by applying Nested Cross Validation for hyperparameter tuning on the architecture.

### Morphological approach:

Rings method :

<div>
  <img src="./images/rm-caudate.png" alt="caudate" width="300" style="margin-right: 10px;" />
  <img src="./images/rm-caudate-separation.png" alt="caudate separation" width="300" />
</div>

<div>
  <img src="./images/rm-cerebellum.png" alt="caudate" width="300" style="margin-right: 10px;" />
  <img src="./images/rm-cerebellum-separation.png" alt="caudate separation" width="300" />
</div>

<div>
  <img src="./images/rm-postcentral.png" alt="caudate" width="300" style="margin-right: 10px;" />
  <img src="./images/rm-postcentral-separation.png" alt="caudate separation" width="300" />
</div>

<div>
  <img src="./images/rm-rostral.png" alt="caudate" width="300" style="margin-right: 10px;" />
  <img src="./images/rm-rostral-separation.png" alt="caudate separation" width="300" />
</div>
