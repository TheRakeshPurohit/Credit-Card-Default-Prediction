import numpy as np
import pandas as pd
from feature_engine.encoding import MeanEncoder


class FeatureEngineering:
    def __init__(self, data) -> None:
        self.data = data

    #     Column name

    # Description

    # customer_id	Represents the unique identification of a customer
    # name	Represents the name of a customer
    # age	Represents the age of a customer ( in years )
    # gender	Represents the gender of a customer( F means Female and M means Male  )
    # owns_car	Represents whether a customer owns a car ( Y means Yes and N means No  )
    # owns_house	Represents whether a customer owns a house ( Y means Yes and N means No  )
    # no_of_children	Represents the number of children of a customer
    # net_yearly_income	Represents the net yearly income of a customer ( in USD )
    # no_of_days_employed	Represents the no of days employed
    # occupation_type	Represents the occupation type of a customer ( IT staff, Managers, Accountants, Cooking staff, etc )
    # total_family_members	Represents the number of family members of a customer
    # migrant_worker	Represents whether a customer is a migrant worker( 1 means Yes and 0 means No  )
    # yearly_debt_payments	Represents the yearly debt payments of a customer  ( in USD )
    # credit_limit	Represents the credit limit of a customer  ( in USD )
    # credit_limit_used(%)	Represents the percentage of credit limit used by a customer
    # credit_score	Represents the credit score of a customer
    # prev_defaults	Represents the number of previous defaults
    # default_in_last_6months	Represents whether a customer has defaulted in the last 6 months ( 1 means Yes and 0 means No  )
    # credit_card_default	Represents whether there will be credit card default  ( 1 means Yes and 0 means No  )

    def add_features(self):

        self.data["coverage_ratio"] = (
            self.data["net_yearly_income"] / self.data["yearly_debt_payments"]
        )

        self.data["precentage_of_number_of_days_employed"] = (
            self.data["no_of_days_employed"] / self.data["no_of_days_employed"].max()
        )
        self.data["precentage_of_number_of_days_employed_per_year"] = (
            self.data["no_of_days_employed"]
            / self.data["no_of_days_employed"].max()
            * 365
        )
        self.data["net_yearly_income_precentage"] = (
            self.data["net_yearly_income"] / self.data["net_yearly_income"].max()
        )
        for i in range(len(self.data["owns_car"])):
            if self.data["owns_car"][i] == "Y" and self.data["owns_house"][i] == "Y":
                self.data["owns_the_car_and_owns_the_home_as_well"] = "Y"
            else:
                self.data["owns_the_car_and_owns_the_home_as_well"] = "N"

            if self.data["owns_car"][i] == "Y" and self.data["owns_house"][i] == "N":
                self.data["owns_the_car_but_does_not_own_the_home"] = "Y"
            else:
                self.data["owns_the_car_but_does_not_own_the_home"] = "N"

            if (
                self.data["migrant_worker"][i] == 1
                and self.data["default_in_last_6months"][i] == 1
            ):
                self.data["migrant_worker_and_default_in_last_6months"] = "Y"
            else:
                self.data["migrant_worker_and_default_in_last_6months"] = "N"
            if (
                self.data["migrant_worker"][i] == 1
                and self.data["default_in_last_6months"][i] == 0
            ):
                self.data["migrant_worker_and_not_default_in_last_6months"] = "Y"
            else:
                self.data["migrant_worker_and_not_default_in_last_6months"] = "N"

            # if (
            #     self.data["occupation_type"][i] == "Laborers"
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["laborer_and_has_previous_defaults"] = "Y"
            # else:
            #     self.data["laborer_and_has_previous_defaults"] = "N"
            # if (
            #     self.data["occupation_type"][i] == "High skill tech staff"
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["high_skill_tech_staff_and_has_previous_defaults"] = "Y"
            # else:
            #     self.data["high_skill_tech_staff_and_has_previous_defaults"] = "N"

            # if (
            #     self.data["occupation_type"][i] == "Managers"
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["managers_and_has_previous_defaults"] = "Y"
            # else:
            #     self.data["managers_and_has_previous_defaults"] = "N"

            # if (
            #     self.data["age"][i] > 18
            #     and self.data["age"][i] <= 25
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["young_adult_and_has_previous_defaults"] = "Y"
            # else:
            #     self.data["young_adult_and_has_previous_defaults"] = "N"
            # if (
            #     self.data["age"][i] > 25
            #     and self.data["age"][i] <= 35
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["adult_and_has_previous_defaults"] = "Y"
            # else:
            #     self.data["adult_and_has_previous_defaults"] = "N"
            # if (
            #     self.data["gender"][i] == "F"
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["Female_that_have_defaults"] = "Y"
            # else:
            #     self.data["Female_that_have_defaults"] = "N"
            # if (
            #     self.data["gender"][i] == "M"
            #     and self.data["has_previous_defaults"][i] == 1
            # ):
            #     self.data["male_that_have_defaults"] = "Y"
            # else:
            #     self.data["male_that_have_defaults"] = "N"

        # if the  number of children is greater than 0, then the number of children is 1
        self.data["has_children"] = self.data["no_of_children"].apply(
            lambda x: 1 if x > 0 else 0
        )

        self.data["credit_score_per_year"] = (
            self.data["credit_score"] / self.data["no_of_days_employed"] * 365
        )

        return self.data

    def mean_encoding(self, train, label_train, x_test, test=True):
        """
        Mean encoding of categorical variables
        """
        cols = [
            "customer_id",
            "name",
            "gender",
            "owns_car",
            "owns_house",
            "occupation_type",
            "owns_the_car_and_owns_the_home_as_well",
            "owns_the_car_but_does_not_own_the_home",
            "migrant_worker_and_default_in_last_6months",
            "migrant_worker_and_not_default_in_last_6months",
        ]
        encoder = MeanEncoder(variables=cols)
        encoder.fit(train, label_train)

        train_t = encoder.transform(train)
        test_t = encoder.transform(x_test)

        return train_t, test_t, encoder
        # "owns_the_car_and_owns_the_home_as_well",
        # "owns_the_car_but_does_not_own_the_home",
        # "does_not_own_the_car_but_owns_the_home",
        # "does_not_own_the_car_and_does_not_own_the_home",
        # "migrant_worker_and_default_in_last_6months",
        # "Female_that_have_defaults",

