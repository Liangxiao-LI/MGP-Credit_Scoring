![💰_Credit_Scoring (5)](https://github.com/BL-Starlord/MGP-Credit_Scoring/assets/81414955/81fda687-babd-4ecd-91b7-55a51175cf0e)

![Static Badge](https://img.shields.io/badge/Contributors-blue?style=plastic&logoColor=blue)

Jemma Moseley, Asiyah Sharif, Liangxiao LI, Binghan Shang, Troy Campbell

![Static Badge](https://img.shields.io/badge/Project_background%20-%20black?style=flat)

Credit scoring is a tool used by lenders (e.g. banks) to help decide whether one qualify for a particular credit card, loan, mortgage or service. The credit score is a number everyone in the UK over 18 is given to indicate their credit worthiness. Each lender uses a different method to calculate their credit scores, which is related to the probability of default. The data set for this project provides information on 150,000 borrowers.

![Static Badge](https://img.shields.io/badge/Project%20Aim%20-%20black?style=flat)

Develop a statistical model/algorithm which accurately predicts the probability of a default on a loan based on data available to investors. Which attributes are more useful in making such predictions?

![Static Badge](https://img.shields.io/badge/Pipeline-Purple)

<p align="center">
<img width="470" alt="Screenshot 2024-02-15 at 12 55 26 PM" src="https://github.com/BL-Starlord/MGP-Credit_Scoring/assets/81414955/053c72f4-5c09-4484-bf27-30f756465a8f">
</p>



![Static Badge](https://img.shields.io/badge/Imputation%20NA-red)

The dataset contains 10 columns, two of which contain NA values: NumberOfDependents contains 3924 NA entries and Monthly Income contains 29731 NA entries. Therefore we implement the imputation method to fill in the NA values based on the random forest model and XGB model. The pipeline is explained in the following image.

<p align="center">
<img width="700" alt="Screenshot 2024-02-14 at 10 14 02 AM" src="https://github.com/BL-Starlord/MGP-Credit_Scoring/assets/81414955/0321e632-63f4-424a-8dca-3e2f355529ee">
</p>
