#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libsvm
# import hyperparameters
import numpy as np
import pandas as pd


# In[2]:


'''
I am setting a global int value for seed here which can be used at multiple places as needed.
If you want to change the seed value you don't have to do it at each and every place. 
You can just change it here.
'''
random_seed = 52


# In[30]:


'''
SVM
'''
def svm(X_train, y_train, epochs, lr, c):
    
    # Setting the seed
    np.random.seed(seed=random_seed)
    
    #dimensions of svm + bias
    dim = X_train.shape[1] 

    # Initializing Weight vector with bias term
    # 2 options
#     w = np.random.uniform(-0.01, 0.01, size = dim)
    w = np.zeros(dim)
    
    for epoch in range(epochs):
        
        # Decaying learning rate
        lr = lr / (1+epoch)
        
        #Shuffle both X and y in the same order
        X_train_s, y_train_s = shuffle_arrays(X_train,y_train)
        
        for x, y in zip(X_train_s, y_train_s):
            
            #Make an update according to mistake done or not
            if (y*np.dot(x,w.T)<=1):
                w = (1-lr)*w + lr*c*y*x
            else:
                w = (1-lr)*w
    return w


# In[4]:


def addOnesColumn(X_train):
    
    # Adding extra feature 1 to the X_matrix
    ones_column = np.ones(X_train.shape[0])
    X_train_b = np.hstack((X_train, ones_column[:, np.newaxis]))
    
    return X_train_b


# In[54]:


def predict(x,w):
    
    temp_output = np.dot(w.T,x)

    if(temp_output >= 0):
        output = 1
    else:
        output = 0
            
    return output


# In[49]:


def get_preds(X, w):
    
    predictions = []
    
    for x in X:
        pred = predict(x, w)
        predictions.append(pred)
        
    return predictions


# In[48]:


def accuracy(X, y, w):
    
    predictions = get_preds(X, w)
    
    acc = np.sum(y == predictions)/len(predictions)
    return acc


# In[7]:


'''
Takes your samples(X) and the labels(y) and shuffle them with the same order
'''
def shuffle_arrays(X, y):
    
    # Setting the seed
    np.random.seed(seed=random_seed)
    
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    
    return X[idx], y[idx]


# In[28]:


def kfold(fold_x_list,fold_y_list, hyperparameters, epochs):
    
#     zipped_folds_list = list(zip(*fold_list))
    
#     fold_x_list = list(zipped_folds_list[0]) ## list of x from each fold
#     fold_y_list = list(zipped_folds_list[1]) ## list of y from each fold
    
#     hyper_parameters = hyperparameters.get_hyperparameters(model='svm') # getting the hyperparameter dict    
    
    # Making the res_df according to the model type
    fold_acc_cols = ['fold1_acc','fold2_acc','fold3_acc','fold4_acc','fold5_acc']    
    cols = fold_acc_cols + list(hyperparameters.keys())
    res_df = pd.DataFrame(columns=cols)
    
    for n, (x_tr, y_tr) in enumerate(zip(fold_x_list,fold_y_list)):
    
        train_x_list = []
        train_y_list = []
        for i in range(0,len(fold_x_list)):
            if(i!=n):
                train_x_list.append(fold_x_list[i])
                train_y_list.append(fold_y_list[i])

        train_x_mat = np.vstack(train_x_list) ## The trainig feature matrix of the current fold
        train_y_mat = np.hstack(train_y_list) ## The training label vector of the current fold

        val_x_mat = x_tr ## The validation feature matrix of the current fold
        val_y_mat = y_tr ## The validation label vector of the current fold
                    
        lr_list = hyperparameters['lr']
        c_list = hyperparameters['C']
            
        for lr in lr_list:
            for c in c_list:
                
                temp_df = pd.DataFrame(columns=cols)
                
                w = svm(train_x_mat, train_y_mat, epochs, lr, c)
                val_acc = accuracy(val_x_mat, val_y_mat, w)
        
                temp_df.loc[0,'lr'] = lr
                temp_df.loc[0,'C'] = c
                temp_df.loc[0,'fold{}_acc'.format(n+1)] = val_acc
                res_df = res_df.append(temp_df)
                

    # Modifying the df            
    ans_df = pd.DataFrame()
    for c in c_list:
        c_df = res_df[res_df['C'] == c]
        c_df1 = c_df.drop(['C'],axis=1)
        a = modifydf(c_df1, lr_list)
        a['C'] = c
        ans_df = ans_df.append(a)
        
    # Calculating the avg accuracy
    ans_df['avg_acc'] = ans_df['fold1_acc'] + ans_df['fold2_acc'] + ans_df['fold3_acc'] + ans_df['fold4_acc'] + ans_df['fold5_acc']
    ans_df['avg_acc'] = ans_df['avg_acc'] / 5
                
    return ans_df.reset_index(drop=True)


# In[9]:


def modifydf(df, lr_list):
    
    ans_df = pd.DataFrame()
    for lr in lr_list:
        a = df[df['lr'] == lr]
        b = a.drop(['lr'],axis=1)
        cols = b.columns
        c = pd.DataFrame(np.diag(b), index=[b.index, b.columns]).T
        c.columns = cols
        c['lr'] = lr
        ans_df = ans_df.append(c)
        
    return ans_df


# In[34]:


def get_CV_splits(X,y,cv=5):
    
    X,y = shuffle_arrays(X,y)
    
    X_list = np.split(X,5)
    y_list = np.split(y,5)
    
    return X_list, y_list


# In[35]:


hyperparameters = {'lr': [1, 0.1, 0.01, 0.001, 0.0001],
 'C': [1, 0.1, 0.01, 0.001, 0.0001]}


# In[57]:


def driver(X,y, epochs = 20):

    # Doing Cross Validation
    X_list, y_list = get_CV_splits(X,y)
    ans_df = kfold(fold_x_list=X_list, fold_y_list=y_list, hyperparameters=hyperparameters, epochs=epochs)
    
    # Getting the best hyperparameters
    best = ans_df[ans_df['avg_acc'] == ans_df['avg_acc'].max()]
    best_lr = best['lr'].values[0]
    best_C = best['C'].values[0]

    # Training the model with best hyperparameters on whole data
    w_best = svm(X, y, epochs, best_lr, best_C)
    preds = get_preds(X, w_best)
    tr_acc = accuracy(X, y, w_best)
    
    return w_best, preds, tr_acc

