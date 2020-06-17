# Build simple AML pipeline end-to-end

### Dataset
https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Fire_Ins_Loss_only.csv

It is a regression problem, target `loss`.

### Task

Build a programm which

1. Reads provided data.

2. Does exploratory data analysis.

For this step you need to distinguish between numeric/categorical/text/date features.
You can keep this step simple and provide a dict with mapping like `column name -> type`.
Ideally type detection is done automatically using different heurictics.

Here is a feature types if you decide not to infer types automatically:
```
loss                           Numeric
crime_burglary                 Numeric
crime_risk                     Numeric
ISO_desc                       Text
Norm_fire_risk                 Numeric
crime_arson                    Numeric
ISO                            Numeric
Weather_risk                   Numeric
Geographical_risk              Numeric
Premium_remain                 Numeric
Renewal_Type                   Categorical
Commercial                     Categorical
crime_property_type            Numeric
Renewal_class                  Categorical
crime_neighbour_watch          Numeric
Previous_claims                Numeric
Exposure                       Numeric
crime_area                     Numeric
ISO_cat                        Categorical
Norm_monthly_rent              Numeric
No_claim_Years                 Numeric
crime_residents                Numeric
Norm_area_m                    Numeric
Rating_Class                   Categorical
Property_size                  Numeric
Residents                      Numeric
crime_community                Numeric
Loan_mortgage                  Numeric
Premium_renew                  Numeric
Sub_Renewal_Class              Categorical
Sub_Rating_Class               Categorical
```

3. (Optional) Does feature enginering.

You don't want to do things like one-hot encoding or normalization here cause it for different models it will de different.
It is more about creating new features like extract day of moth/month of year from dates;
or removing potentially useless features like IDs.

4. Does CV partitioning (keep 10-20% for holdout).

5. Builds a pipeline which can train model.

As we discussed during the class, pipeline should be able to preprocess features (encoding, imputation, normalization),
and fit a model.

This part can be very simple like
```
categorical  ->  one-hot encoding  ->                 ->
numeric      ->  imputation        -> standarization  -> linear regression
text         ->  tf-idf            ->                 ->
```

You can build multiple pipelines: for linear regression and tree based model like Random Forest.
But single model pipeline also works.
*Note: lookout for preprocessing for linear models vs trees*

5. Optimizes hyperparameters of models.
Use GridSearch, RandomSearch or Bayesian optimization.

6. Reports scores of models.

### To submit the solution
1. Fork https://github.com/vikua/aml-class-20
2. Create a directory with your name in `homework`
4. Put your solution there
5. Create a PR from your fork into https://github.com/vikua/aml-class-20 repository

Alternatively you can create a zip file with your solution and send it to me.

Try to keep dependencies in `requrements.txt` file so that results are easy to reproducible.

### (Optional) If you up for a challenge!
This task can be made as simple or as complex as you want it to.

Some ideas:
1. Add model diversity - train linear, trees/gbm, neural nets, etc. and build a "leaderboard" based on the metric you picked.
2. Try to throw another datasets into your program.
Can it handle and train good models for another regressioon dataset - https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Pred_Main_Reg.csv (target `qty_replaced`)?
Cat it handle binary classification problem dataset - https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club.csv (target `is_bad`) ?


Columns of https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Pred_Main_Reg.csv
```
qty_replaced                   Numeric, Target
has_matlspecs_1                Boolean
weight                         Numeric
material_type                  Categorical
has_documents_1                Boolean
area4                          Numeric
material_type_1                Categorical
has_weldspecs_1                Boolean
area1                          Numeric
area2                          Numeric
area3                          Numeric
has_coatings                   Numeric
has_qspecs                     Boolean
has_matlspecs                  Boolean
surface_matl_1                 Boolean
surface_matl                   Boolean
Date                           Date
has_documents                  Boolean
has_weldspecs                  Numeric
part_desc                      Text
m_weight                       Numeric
has_materialtype               Boolean
material_group_1               Categorical
material_group                 Categorical
has_qspecs_1                   Boolean
material_id                    Categorical
rig_plant                      Categorical
has_coatings_1                 Boolean
```

https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Lending_Club.csv
```
Id                             Numeric
is_bad                         Numeric, 1 or 0 Target
emp_title                      Text
emp_length                     Numeric
home_ownership                 Categorical
annual_inc                     Numeric
verification_status            Categorical
paymnt_plan                    Categorical
Notes                          Text
Purpose                        Text
Purpose_cat                    Categorical
zip_code                       Categorical
addr_state                     Categorical
debt-to-income                 Numeric
delinq_2yrs                    Numeric
earliest_cr_line               Date
inq_last_6mths                 Numeric
mths_since_last_delinq         Numeric
mths_since_last_record         Numeric
open_acc                       Numeric
pub_rec                        Numeric
revol_bal                      Numeric
revol_util                     Numeric
total_acc                      Numeric
initial_list_status            Categorical
collections_12_mths_ex_med     Numeric
mths_since_last_major_derog    Numeric
policy_code                    Categoricals
```