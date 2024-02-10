
![💰_Credit_Scoring (5)](https://github.com/BL-Starlord/MGP-Credit_Scoring/assets/81414955/81fda687-babd-4ecd-91b7-55a51175cf0e)

![Static Badge](https://img.shields.io/badge/Project_background%20-%20black?style=flat)

Credit scoring is a tool used by lenders (e.g. banks) to help decide whether one qualifies for a particular credit card, loan, mortgage or service. The credit score is a number everyone in the UK over 18 is given to indicate their credit worthiness. Each lender uses a different method to calculate their credit scores, which is related to the probability of default. The data set for this project provides information on 150,000 borrowers.

![Static Badge](https://img.shields.io/badge/Project%20Aim%20-%20black?style=flat)

Develop a statistical model/algorithm which accurately predicts the probability of a default on a loan based on data available to investors. Which attributes are more useful in making such predictions?

![Static Badge](https://img.shields.io/badge/Imputation%20NA-red)

The dataset contains 10 columns, two of which contain NA values: NumberOfDependents contains 3924 NA entries and Monthly Income contains 29731 NA entries. Therefore we implement the imputation method to fill in the NA values based on the random forest model and XGB model. The pipeline is explained in the following image.

<img width="1000" alt="Screenshot 2024-02-10 at 11 17 24 PM" src="https://github.com/BL-Starlord/MGP-Credit_Scoring/assets/81414955/675fb558-4790-45ef-9e67-572a4b1f2952">
