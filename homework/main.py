from pipeline.params_estimator import ParamsEstimator

types_dict = {
    'crime_burglary': 'numeric',
    'crime_risk': 'numeric',
    'ISO_desc': 'text',
    'Norm_fire_risk': 'numeric',
    'crime_arson': 'numeric',
    'ISO': 'numeric',
    'Weather_risk': 'numeric',
    'Geographical_risk': 'numeric',
    'Premium_remain': 'numeric',
    'Renewal_Type': 'categorical',
    'Commercial': 'categorical',
    'crime_property_type': 'numeric',
    'Renewal_class': 'categorical',
    'crime_neighbour_watch': 'numeric',
    'Previous_claims': 'numeric',
    'Exposure': 'numeric',
    'crime_area': 'numeric',
    'ISO_cat': 'categorical',
    'Norm_monthly_rent': 'numeric',
    'No_claim_Years': 'numeric',
    'crime_residents': 'numeric',
    'Norm_area_m': 'numeric',
    'Rating_Class': 'categorical',
    'Property_size': 'numeric',
    'Residents': 'numeric',
    'crime_community': 'numeric',
    'Loan_mortgage': 'numeric',
    'Premium_renew': 'numeric',
    'Sub_Renewal_Class': 'categorical',
    'Sub_Rating_Class': 'categorical'
}


def main():
    url = 'https://s3.amazonaws.com/datarobot_public_datasets/DR_Demo_Fire_Ins_Loss_only.csv'
    explanatory = types_dict.keys()
    target = 'loss'
    n_folds = 4
    random_state = 42
    split_ratio = .1
    scoring = 'neg_mean_absolute_error'

    parest = ParamsEstimator(
        url=url,
        explanatory=explanatory,
        target=target,
        scoring=scoring,
        types_dict=types_dict,
        split_ratio=split_ratio,
        n_folds=n_folds,
        random_state=random_state
    )

    parest.fit()


if __name__ == '__main__':
    main()