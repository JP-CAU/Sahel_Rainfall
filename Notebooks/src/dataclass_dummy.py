# Imports this class requires
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


# Loads and prepares the data.
# Data can be accessed over class variables: train_input_FOCI, train_target_FOCI, validation_input_FOCI...
# Unprocessed data is saved. When seq_length or feature_list are None, they are not used.
# Data is processed to have 0 mean and 1 std.
# possible_features variable contains list with all feature names, even those that have not been selected.
# For lead_time = 0, the sequences currently DO NOT contain previous prec_sahel values.
# flatten_seq = False means that each sample consits of the arrays for each time step (e.g. [t1, t2, ...])
# flatten_seq = True means that each sample contains all the values of the time steps. (array containing values vs array containing arrays of values)
class Data:
    # Split = [train,validation, test], lead_time >= 0, seq_length > 0 or None, feature_list None or list of feature names
    def __init__(
        self,
        split=[0.9, 0.0, 0.1],
        lead_time=0,
        seq_length=None,
        flatten_seq=False,
        feature_list=None,
    ):
        # Variables
        self.feature_list = feature_list
        self.possible_features = (
            []
        )  # Filled later on to track all possible features before selection
        self.__data_url = (
            "https://github.com/MarcoLandtHayen/climate_index_collection/"
            "releases/download/v2023.03.29.1/climate_indices.csv"
        )
        self.lead_time = lead_time

        # Load the data
        FOCI, CESM = self.__load_data()

        # Split into input and target
        input_FOCI, target_FOCI = self.__get_input_target(FOCI, lead_time)
        input_CESM, target_CESM = self.__get_input_target(CESM, lead_time)

        # Save possible features that can be selected
        self.possible_features = list(input_FOCI.columns)

        # Select features if any are set
        # print(input_FOCI.shape)
        # input_FOCI = input_FOCI if feature_list is None else input_FOCI.drop(columns=[col for col in self.possible_features if col not in feature_list])
        # input_CESM = input_CESM if feature_list is None else input_CESM.drop(columns=[col for col in self.possible_features if col not in feature_list])
        # print(input_FOCI.shape)
        # Split into train, validation and test, based on split percentage given
        (
            self.train_input_FOCI,
            self.validation_input_FOCI,
            self.test_input_FOCI,
        ) = self.__get_train_validation_test(input_FOCI, split)
        (
            self.train_target_FOCI,
            self.validation_target_FOCI,
            self.test_target_FOCI,
        ) = self.__get_train_validation_test(target_FOCI, split)

        (
            self.train_input_CESM,
            self.validation_input_CESM,
            self.test_input_CESM,
        ) = self.__get_train_validation_test(input_CESM, split)
        (
            self.train_target_CESM,
            self.validation_target_CESM,
            self.test_target_CESM,
        ) = self.__get_train_validation_test(target_CESM, split)

        # Fit standardscaler on trainings data and then scale train, validation and test input with it to obtain 0 mean and 1 std
        # Returned values are numpy arrays
        (
            self.train_input_FOCI,
            self.validation_input_FOCI,
            self.test_input_FOCI,
        ) = self.__scale_data_input(
            self.train_input_FOCI, self.validation_input_FOCI, self.test_input_FOCI
        )
        (
            self.train_input_CESM,
            self.validation_input_CESM,
            self.test_input_CESM,
        ) = self.__scale_data_input(
            self.train_input_CESM, self.validation_input_CESM, self.test_input_CESM
        )

        # Do the same with the target data
        (
            self.train_target_FOCI,
            self.validation_target_FOCI,
            self.test_target_FOCI,
        ) = self.__scale_data_target(
            self.train_target_FOCI, self.validation_target_FOCI, self.test_target_FOCI
        )
        (
            self.train_target_CESM,
            self.validation_target_CESM,
            self.test_target_CESM,
        ) = self.__scale_data_target(
            self.train_target_CESM, self.validation_target_CESM, self.test_target_CESM
        )

        # Turn target arrays into 1D arrays, reshape only if array actually exists
        # otherwise an error would be thrown.
        if len(self.train_target_FOCI) > 0:
            self.train_target_FOCI = self.train_target_FOCI.reshape(-1)
            self.train_target_CESM = self.train_target_CESM.reshape(-1)
        if len(self.validation_target_FOCI) > 0:
            self.validation_target_FOCI = self.validation_target_FOCI.reshape(-1)
            self.validation_target_CESM = self.validation_target_CESM.reshape(-1)
        if len(self.test_target_FOCI) > 0:
            self.test_target_FOCI = self.test_target_FOCI.reshape(-1)
            self.test_target_CESM = self.test_target_CESM.reshape(-1)

        # Turn data into sequences consisting of seq_length timesteps
        if seq_length is not None:
            # Check for valid seq_length
            if seq_length <= 0:
                raise ValueError(
                    "seq_length has to be an integer > 0 or None but is{}".format(
                        seq_length
                    )
                )

            # Train FOCI
            if len(self.train_input_FOCI) > 0:
                self.train_input_FOCI, self.train_target_FOCI = self.__into_sequence(
                    self.train_input_FOCI,
                    self.train_target_FOCI,
                    seq_length,
                    flatten_seq,
                )
            # Validation FOCI
            if len(self.validation_input_FOCI) > 0:
                (
                    self.validation_input_FOCI,
                    self.validation_target_FOCI,
                ) = self.__into_sequence(
                    self.validation_input_FOCI,
                    self.validation_target_FOCI,
                    seq_length,
                    flatten_seq,
                )
            # Test FOCI
            if len(self.test_input_FOCI) > 0:
                self.test_input_FOCI, self.test_target_FOCI = self.__into_sequence(
                    self.test_input_FOCI, self.test_target_FOCI, seq_length, flatten_seq
                )
            # Train CESM
            if len(self.train_input_CESM) > 0:
                self.train_input_CESM, self.train_target_CESM = self.__into_sequence(
                    self.train_input_CESM,
                    self.train_target_CESM,
                    seq_length,
                    flatten_seq,
                )
            # Validation CESM
            if len(self.validation_input_CESM) > 0:
                (
                    self.validation_input_CESM,
                    self.validation_target_CESM,
                ) = self.__into_sequence(
                    self.validation_input_CESM,
                    self.validation_target_CESM,
                    seq_length,
                    flatten_seq,
                )
            # Test CESM
            if len(self.test_input_CESM) > 0:
                self.test_input_CESM, self.test_target_CESM = self.__into_sequence(
                    self.test_input_CESM, self.test_target_CESM, seq_length, flatten_seq
                )

    # Loads the data from url or disk, returns FOCI, CESM as pandas dataframes.
    def __load_data(self):
        # Check if data exists on disk, if so load from disk, otherwise from url
        if os.path.exists("climate_indices.csv"):
            climind = pd.read_csv("climate_indices.csv")
        else:
            climind = pd.read_csv(self.__data_url)
            # Save data to disk
            climind.to_csv("climate_indices.csv", index=False)

        # Split into FOCI and CESM
        climind = climind.set_index(["model", "year", "month", "index"]).unstack(
            level=-1
        )["value"]

        if self.feature_list is not None:
            climind.drop(
                columns=[
                    col
                    for col in climind.columns
                    if col not in self.feature_list
                ],
                inplace=True
            )

        one_hot_encoded = pd.get_dummies(
            climind.index.codes[2], prefix="month", dtype=int
        )

        for i in range(12):
            climind["month_" + str(i)] = list(one_hot_encoded["month_" + str(i)])

        FOCI = climind.loc[("FOCI")].reset_index().drop(columns=["year", "month"])
        CESM = climind.loc[("CESM")].reset_index().drop(columns=["year", "month"])

        # Return them
        return FOCI, CESM

    # Takes the FOCI or CESM pandas dataframe and returns the input and target
    # Input and target depends on the lead_time, for lead_time >0 input contains
    # PREC_SAHEL of the current time step, for lead_time = 0 it does not.
    # lead_time determines how many months in advance the target is.
    def __get_input_target(self, data, lead_time):
        # Check if lead_time is vlaid
        if lead_time < 0:
            raise ValueError(
                "lead_time has to have a value >= 0 but has value {}".format(
                    self.__lead_time
                )
            )

        # Split into target and input, input has to omit the last lead_time elements or there would be no target for them
        target = data.loc[:, data.columns == "PREC_SAHEL"]
        input = (
            data.loc[:, data.columns != "PREC_SAHEL"]
            if lead_time == 0
            else data[:-lead_time]
        )

        # Adjust target for lead_time if needed
        if lead_time > 0:
            target = target[lead_time:]

        # Return input and target
        return input, target

    # Splits data based on the given split into train, validation and test
    # split = [train, validation, test] as decimal indicating percentage
    def __get_train_validation_test(self, data, split):
        # Check if split is valid
        if sum(split) != 1 or split[0] <= 0 or any(i < 0 for i in split):
            raise ValueError(
                "Invalid split has been passed. Values can be negative, have to sum up to 1 and train has to be > 0"
            )

        # Get number of samples for each split
        n_train = int(split[0] * len(data))
        n_val = int(split[1] * len(data))
        n_test = int(split[2] * len(data))  # Only used to check if there is a test set

        # Create the splits
        train = data[:n_train] if n_train > 0 else []
        val = data[n_train : n_train + n_val] if n_val > 0 else []
        test = data[n_train + n_val :] if n_test > 0 else []

        train = np.array(train)
        val = np.array(val)
        test = np.array(test)

        # Return them
        return train, val, test

    # Scales the data to have mean of 0 and std of 1
    # only fits on the training input data
    def __scale_data_input(self, train, val, test):
        scaler = StandardScaler()
        if len(train) > 0:
            scaler.fit(train[:, :-12])
        else:
            scaler.fit(train)

        # Transform train, val, test input to have 0 mean and 1std
        # only perform transformation if set is non empty

        if self.feature_list is not None:
            scaled_train = scaler.transform(train[:, :-12]) if len(train) > 0 else []
            if len(train) > 0:
                scaled_train = np.concatenate(
                    (scaled_train, train[:, len(self.feature_list) :]), axis=1
                )

            scaled_val = scaler.transform(val[:, :-12]) if len(val) > 0 else []
            if len(val) > 0:
                scaled_val = np.concatenate(
                    (scaled_val, val[:, len(self.feature_list) :]), axis=1
                )

            scaled_test = scaler.transform(test[:, :-12]) if len(test) > 0 else []
            if len(test) > 0:
                scaled_test = np.concatenate(
                    (scaled_test, test[:, len(self.feature_list) :]), axis=1
                )

        else:
            scaled_train = scaler.transform(train[:, :-12]) if len(train) > 0 else []
            if len(train) > 0:
                scaled_train = (
                    np.concatenate((scaled_train, train[:, 29:]), axis=1)
                    if self.lead_time > 0
                    else np.concatenate((scaled_train, train[:, 28:]), axis=1)
                )

            scaled_val = scaler.transform(val[:, :-12]) if len(val) > 0 else []
            if len(val) > 0:
                scaled_val = (
                    np.concatenate((scaled_val, val[:, 29:]), axis=1)
                    if self.lead_time > 0
                    else np.concatenate((scaled_val, val[:, 28:]), axis=1)
                )

            scaled_test = scaler.transform(test[:, :-12]) if len(test) > 0 else []
            if len(test) > 0:
                scaled_test = (
                    np.concatenate((scaled_test, test[:, 29:]), axis=1)
                    if self.lead_time > 0
                    else np.concatenate((scaled_test, test[:, 28:]), axis=1)
                )

        return scaled_train, scaled_val, scaled_test

    def __scale_data_target(self, train, val, test):
        scaler = StandardScaler()
        scaler.fit(train)

        # Transform train, val, test input to have 0 mean and 1std
        # only perform transformation if set is non empty
        scaled_train = scaler.transform(train) if len(train) > 0 else []
        scaled_val = scaler.transform(val) if len(val) > 0 else []
        scaled_test = scaler.transform(test) if len(test) > 0 else []

        return scaled_train, scaled_val, scaled_test

    # Turns input into a sequence consisting of seq_length time steps
    # and selects i+seq_length-1 as the corresponding target index for the i'th sequence.
    def __into_sequence(self, input, target, seq_length, flatten_seq):
        input_seq = np.array(
            [input[i : i + seq_length] for i in range(len(input) - seq_length)]
        )
        target_seq = np.array(
            [target[i + seq_length - 1] for i in range(len(target) - seq_length)]
        )

        # Flatten sample containing the sequences if wanted (sample = [t_0, t_1,...] with t_0 = [feature_1, feature_2,...])
        if flatten_seq:
            input_seq = [seq.reshape(-1) for seq in input_seq]

        return input_seq, target_seq
