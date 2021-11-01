import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *


def main():
    """
    ML Pipeline for our best submission.

    It will split the data in three parts according to the
    number of jets, process each of these bins and train a 
    classifier for each of them. Then it will make the
    predictions on the test data and output a .csv file with
    the results.

    """
    ### Load Training Data
    DATA_TRAIN_PATH = '../data/train.csv'
    y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    ### Load Test Data 
    DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    OUTPUT_PATH = "results.csv"
    ### Test Data Partition
    jet_num_col_id = 22
    no_jet_test_ids = np.where(tX_test[:,jet_num_col_id] == 0)[0]
    one_jet_test_ids = np.where(tX_test[:,jet_num_col_id] == 1)[0]
    multi_jet_test_ids = np.where(tX_test[:,jet_num_col_id] > 1)[0]
    ### Data processing pipeline
    feature_1_mean_tr = (tX_train[:,0][tX_train[:,0] != -999]).mean()
    training_sets, ys_tr = data_pipeline(y_train,tX_train,feature_1_mean_tr,stage=2)
    test_sets, _ = data_pipeline([],tX_test,feature_1_mean_tr,stage=2)
    ###
    no_jet_tr = training_sets[0]
    one_jet_tr = training_sets[1]
    multi_jet_tr = training_sets[2]
    ###
    no_jet_te = test_sets[0]
    one_jet_te = test_sets[1]
    multi_jet_te = test_sets[2]
    ### Standardize the features
    no_jet_tr, no_jet_tr_mean, no_jet_tr_std = standardize(no_jet_tr)
    one_jet_tr, one_jet_tr_mean, one_jet_tr_std = standardize(one_jet_tr)
    multi_jet_tr, multi_jet_tr_mean, multi_jet_tr_std = standardize(multi_jet_tr)
    ###
    no_jet_te, _, _ = standardize(no_jet_te,no_jet_tr_mean,no_jet_tr_std)
    one_jet_te, _, _ = standardize(one_jet_te,one_jet_tr_mean,one_jet_tr_std)
    multi_jet_te, _, _ = standardize(multi_jet_te,multi_jet_tr_mean,multi_jet_tr_std)
    ### Model Training
    w_no_jet, w_one_jet, w_multi_jet = model_training(ys_tr,[no_jet_tr,one_jet_tr,multi_jet_tr],model="least_squares")
    ### Prediction
    no_jet_pred = predict_label(no_jet_te,w_no_jet)
    one_jet_pred = predict_label(one_jet_te,w_one_jet)
    multi_jet_pred = predict_label(multi_jet_te,w_multi_jet)
    predictions = np.zeros((tX_test.shape[0],1))
    predictions[no_jet_test_ids] = np.reshape(no_jet_pred, (no_jet_pred.shape[0],1))
    predictions[one_jet_test_ids] = np.reshape(one_jet_pred, (one_jet_pred.shape[0],1))
    predictions[multi_jet_test_ids] = np.reshape(multi_jet_pred, (multi_jet_pred.shape[0],1))
    ### Output
    create_csv_submission(ids_test, predictions, OUTPUT_PATH)
    print("Done")


def predict_label(x,weights,model="least_squares"):
    """Predict the labels based on the rule dicted by the model selection."""
    pred = x@weights
    if model not in ["reg_logistic_regression","logistic_regression"]:
        pred[pred < 0] = -1
        pred[pred >= 0] = 1
    else:
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
    return pred

def model_training(ys,training_sets, model="least_squares",lambda_=0.0001,gamma=5e-9,max_iter=1000):
    """ Train the given model on the data provided."""
    no_jet_training_set = training_sets[0]
    one_jet_training_set = training_sets[1]
    multi_jet_training_set = training_sets[2]
    ###
    no_jet_y = np.reshape(ys[0],(ys[0].shape[0],1))
    one_jet_y = np.reshape(ys[1],(ys[1].shape[0],1))
    multi_jet_y = np.reshape(ys[2],(ys[2].shape[0],1))
    ### 
    w_init_1 = np.zeros((no_jet_training_set.shape[1],1))
    w_init_2 = np.zeros((one_jet_training_set.shape[1],1))
    w_init_3 = np.zeros((multi_jet_training_set.shape[1],1))
    ###
    if model == "least_squares":
        w_1, _ = least_squares(no_jet_y,no_jet_training_set)
        w_2, _ = least_squares(one_jet_y,one_jet_training_set)
        w_3, _ = least_squares(multi_jet_y,multi_jet_training_set)
    elif model == "ridge_regression":
        w_1, _ = ridge_regression(no_jet_y,no_jet_training_set,lambda_)
        w_2, _ = ridge_regression(one_jet_y,one_jet_training_set,lambda_)
        w_3, _ = ridge_regression(multi_jet_y,multi_jet_training_set,lambda_)
    elif model == "SGD":
        w_1, _ = least_squares_SGD(no_jet_y,no_jet_training_set,w_init_1,max_iter,gamma)
        w_2, _ = least_squares_SGD(one_jet_y,one_jet_training_set,w_init_2,max_iter,gamma)
        w_3, _ = least_squares_SGD(multi_jet_y,multi_jet_training_set,w_init_3,max_iter,gamma)
    elif model == "GD":
        w_1, _ = least_squares_GD(no_jet_y,no_jet_training_set,w_init_1,max_iter,gamma)
        w_2, _ = least_squares_GD(one_jet_y,one_jet_training_set,w_init_2,max_iter,gamma)
        w_3, _ = least_squares_GD(multi_jet_y,multi_jet_training_set,w_init_3,max_iter,gamma)
    elif model == "logistic_regression": 
        w_1, _ = logistic_regression(no_jet_y,no_jet_training_set,w_init_1,max_iter,gamma)
        w_2, _ = logistic_regression(one_jet_y,one_jet_training_set,w_init_2,max_iter,gamma)
        w_3, _ = logistic_regression(multi_jet_y,multi_jet_training_set,w_init_3,max_iter,gamma)
    else: # regularized logistic regression
        w_1, _ = reg_logistic_regression(no_jet_y,no_jet_training_set,lambda_,w_init_1,max_iter,gamma)
        w_2, _ = reg_logistic_regression(one_jet_y,one_jet_training_set,lambda_,w_init_2,max_iter,gamma)
        w_3, _ = reg_logistic_regression(multi_jet_y,multi_jet_training_set,lambda_,w_init_3,max_iter,gamma)
    return w_1, w_2, w_3

def data_pipeline(y,tX,training_mean,stage=0):
    """ Process the data according to our pipeline, c.f. the readme."""
    ### Do mean imputation for a feature with undefined values
    tX[:,0][tX[:,0] == -999] = training_mean
    ### Split the data in three bins
    labels_set = []
    if any(y): 
        labels_set = split_labels(y,tX)
    no_jet_data_set, one_jet_data_set, multi_jet_data_set = split_in_bins(tX)
    if stage >=1 :
        ### Feature Selection 
        no_jet_data_set = no_jet_data_set[:,[0,1,2,3,4,6,7,8,9,12,13,15]] 
        one_jet_data_set = one_jet_data_set[:,[0,1,2,3,4,5,6,7,8,9,12,17,18,19]] 
        multi_jet_data_set = multi_jet_data_set[:,[1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21,22,26,27,29]] 
    ### Add dummy feature
    no_jet_data_set = np.concatenate((np.ones((no_jet_data_set.shape[0],1)),no_jet_data_set), axis=1)
    one_jet_data_set = np.concatenate((np.ones((one_jet_data_set.shape[0],1)),one_jet_data_set), axis=1)
    multi_jet_data_set = np.concatenate((np.ones((multi_jet_data_set.shape[0],1)),multi_jet_data_set), axis=1)
    if stage >= 2:
        ### Add joint features 
        no_jet_data_set, one_jet_data_set, multi_jet_data_set = add_joint_features(no_jet_data_set, 
                                                                                    one_jet_data_set, multi_jet_data_set)
        ### Add polynomial features
        no_jet_data_set, one_jet_data_set, multi_jet_data_set = add_polynomial_features(no_jet_data_set, 
                                                                                    one_jet_data_set, multi_jet_data_set)
    features_set = [no_jet_data_set, one_jet_data_set, multi_jet_data_set]
    return features_set, labels_set 
    
def add_polynomial_features(no_jet_data_set, one_jet_data_set, multi_jet_data_set):
    """Expand the features of each bins with the corresponding degree."""
    ### Each lists determines the optimal degree of each feature
    no_jet_col_best_degree = [1,2,3,1,1,2,2,2,1,2,1,2,1]
    one_jet_col_best_degree = [1,2,1,2,2,2,1,2,1,1,2,2,2,2,2]
    multi_jet_col_best_degree = [1,1,2,2,2,3,2,2,2,2,1,1,2,1,2,1,1,1,1,1,1,]
    ###
    no_jet_data_set = col_poly_expansion(no_jet_data_set,no_jet_col_best_degree)
    one_jet_data_set = col_poly_expansion(one_jet_data_set,one_jet_col_best_degree)
    multi_jet_data_set = col_poly_expansion(multi_jet_data_set,multi_jet_col_best_degree)
    return no_jet_data_set, one_jet_data_set, multi_jet_data_set

def add_joint_features(no_jet_data_set, one_jet_data_set, multi_jet_data_set):
    """Add new features for each bins based on the best product of features."""
    ### Each lists determines the top 6 optimal joint product of features
    no_jet_best_joint_indices = [(1, 6),(6, 11),(6, 9),(5, 6),(2, 4),(1, 11)]
    one_jet_best_joint_indices = [(4, 11),(4, 6),(4, 12),(3, 4),(4, 9),(0, 2)]
    multi_jet_best_joint_indices = [(1, 6), (3, 10), (4, 10), (5, 10), (3, 6), (2, 6)]
    ###
    no_jet_data_set = joint_expansion(no_jet_data_set, no_jet_best_joint_indices)
    one_jet_data_set = joint_expansion(one_jet_data_set, one_jet_best_joint_indices)
    multi_jet_data_set = joint_expansion(multi_jet_data_set, multi_jet_best_joint_indices)
    return no_jet_data_set, one_jet_data_set, multi_jet_data_set
    
def split_labels(y,tX):
    """Split the y labels in three bins according to the number of jets."""
    jet_num_col_id = 22
    no_jet_indices = np.where(tX[:,jet_num_col_id] == 0)[0]
    one_jet_indices = np.where(tX[:,jet_num_col_id] == 1)[0]
    multi_jet_indices = np.where(tX[:,jet_num_col_id] > 1)[0]
    ###
    no_jet_y = y[no_jet_indices]
    one_jet_y = y[one_jet_indices]
    multi_jet_y = y[multi_jet_indices]
    labels_set = [no_jet_y,one_jet_y,multi_jet_y]
    return labels_set

def split_in_bins(tX):
    """Split the data in bins according the the number of jets."""
    jet_num_col_id = 22
    valid_features_no_jet = [0,1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21]
    valid_features_one_jet = [0,1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,29]
    ###
    no_jet_indices = np.where(tX[:,jet_num_col_id] == 0)[0]
    one_jet_indices = np.where(tX[:,jet_num_col_id] == 1)[0]
    multi_jet_indices = np.where(tX[:,jet_num_col_id] > 1)[0]
    ###
    no_jet_data_set = tX[no_jet_indices,:]
    one_jet_data_set = tX[one_jet_indices,:]
    multi_jet_data_set = tX[multi_jet_indices,:]
    ### Filtering unvalid entries
    no_jet_data_set = no_jet_data_set[:,valid_features_no_jet]
    one_jet_data_set = one_jet_data_set[:,valid_features_one_jet]
    return no_jet_data_set, one_jet_data_set, multi_jet_data_set

def joint_expansion(x,indices):
    """Add the joint features expansion for x at the given indices."""
    for i,j in indices:
        new_feature = np.reshape(x[:,i]*x[:,j],(x.shape[0],1))
        x = np.concatenate((x,new_feature),axis=1)
    return x 

def col_poly_expansion(tX,degree_list):
    """Add the polynomial expansion of the features of tX according to the given degrees."""
    for col_id, degree in enumerate(degree_list):
        if degree > 1: # Avoid duplicated features
            new_feature =  np.reshape(np.power(tX[:,col_id], degree),(tX.shape[0],1))
            if (new_feature > 0).sum() == new_feature.shape[0]:
                new_feature = np.log(new_feature)
            tX = np.concatenate((tX,new_feature),axis=1)
    return tX

def standardize(x,mean=[],std=[]):
    """Standardize the data either based on the given mean and std or on its own mean and std."""
    x_rand = x[:,1:] # The first column is a dummy feature that we cannot/should not normalize
    if not any(mean) and not any(std):
        mean = x_rand.mean(axis=0)
        std = x_rand.std(axis=0)
    x_rand_norm = (x_rand-mean)/std
    x_cte = np.reshape(x[:,0],(x.shape[0],1))
    x = np.concatenate((x_cte,x_rand_norm),axis=1)
    return x, mean, std

if __name__ == "__main__":
    main()