
# coding: utf-8

# In[29]:


import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='8145'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    return train0, train1, test0, test1
    pass


# In[30]:


def task_1_feature_extraction(train_data_set):
    #feature extraction i.e feature1 = Mean of brightness and feature2 = Standard Deviation of brightness
    
    #print("\n task_1_feature_extraction running")
    #print(len(train_data_set[0]))
    
    n = len(train_data_set)
    image_feature_matrix = []
    for i in range(n):
        image_feature_matrix.append([])
        
    #feature1 = Mean of brightness calculation
    for i in range(n):
        train_array = []
        row_len = len(train_data_set[i])
        for j in range(row_len):
            for k in range(row_len):
                train_array.append(train_data_set[i][j][k])
        
        train_mean = numpy.mean(train_array)
        image_feature_matrix[i].append(train_mean)
    
    
    
    #feature2 = Standard Deviation of brightness
    for i in range(n):
        train_array = []
        row_len = len(train_data_set[i])
        for j in range(row_len):
            for k in range(row_len):
                train_array.append(train_data_set[i][j][k])
                
        train_sd = numpy.std(train_array)
        image_feature_matrix[i].append(train_sd)
        
    #print("\n Feature Extract of Images: ",image_feature_matrix) 
    #print("\n task_1_feature_extraction completed")

    return image_feature_matrix
    
    
    


# In[31]:


def mean_and_variance_feature1_calulation(image_feature_matrix):
    feature1 = []
    for i in range(len(image_feature_matrix)):
        feature1.append(image_feature_matrix[i][0])
        
    mean_feature1 = numpy.mean(feature1)
    #print("\n mean of feature1 : ",mean_feature1)
    
    variance_feature1 = numpy.var(feature1)
    #print("\n variance of feature1 : ",variance_feature1)
    
    return mean_feature1, variance_feature1
    
    
def mean_and_variance_feature2_calulation(image_feature_matrix):
    feature2 = []
    for i in range(len(image_feature_matrix)):
        feature2.append(image_feature_matrix[i][1])
        
    mean_feature2 = numpy.mean(feature2)
    #print("\n mean of feature2 : ",mean_feature2)
    
    variance_feature2 = numpy.var(feature2)
    #print("\n variance of feature2 : ",variance_feature2)
    
    return mean_feature2, variance_feature2


# In[32]:


def label_prediction(test_image_feature_matrix, mean_feature1_digit0, variance_feature1_digit0, mean_feature2_digit0, variance_feature2_digit0,
                    mean_feature1_digit1, variance_feature1_digit1, mean_feature2_digit1, variance_feature2_digit1):
    #P(X|Y=0) = P(X1|Y=0)*P(X2|Y=0)*0.5 --> probability of digit0 #
    
    n = len(test_image_feature_matrix)
    label_prediction = []
    
    for i in range(n):
        feature1_digit0_prob = conditional_probability_calulcation(test_image_feature_matrix[i][0], mean_feature1_digit0, math.sqrt(variance_feature1_digit0))
        
        feature2_digit0_prob = conditional_probability_calulcation(test_image_feature_matrix[i][1], mean_feature2_digit0, math.sqrt(variance_feature2_digit0))

        digit0_prob = feature1_digit0_prob*feature2_digit0_prob*0.5
        
        
        feature1_digit1_prob = conditional_probability_calulcation(test_image_feature_matrix[i][0], mean_feature1_digit1, math.sqrt(variance_feature1_digit1))
        
        feature2_digit1_prob = conditional_probability_calulcation(test_image_feature_matrix[i][1], mean_feature2_digit1, math.sqrt(variance_feature2_digit1))

        digit1_prob = feature1_digit1_prob*feature2_digit1_prob*0.5
        
        
        #print(digit0_prob,digit1_prob)
        
        if digit0_prob >= digit1_prob:
            label = 0
        else:
            label = 1
            
        label_prediction.append(label)
        
    return label_prediction


        
def conditional_probability_calulcation(x,nu,sigma):
    term1 = 1/(sigma*math.sqrt(2*math.pi))
    term2 = math.exp(-(math.pow((x-nu), 2))/(2*math.pow(sigma,2)))
    return term1*term2



# In[47]:


def digit0_accuracy_prediction(digit0_label_prediction):
    n = len(digit0_label_prediction)
    
    count = 0
    for i in range(n):
        if digit0_label_prediction[i] == 0:
            count+=1
            
    digit0_accuracy = count/n
    
    return digit0_accuracy
            

def digit1_accuracy_prediction(digit1_label_prediction):
    n = len(digit1_label_prediction)
    
    count = 0
    for i in range(n):
        if digit1_label_prediction[i] == 1:
            count+=1
            
    digit1_accuracy = count/n
    
    return digit1_accuracy


# In[48]:


if __name__ == '__main__':
    #I'm Jeevan Gupta (ASU ID: 1223328145)
    
    train0, train1, test0, test1 = main()
    
    train0_image_feature_matrix = task_1_feature_extraction(train0)
    #print(train0_image_feature_matrix[0])
    
    mean_feature1_digit0, variance_feature1_digit0 = mean_and_variance_feature1_calulation(train0_image_feature_matrix)
    print("\n mean_feature1_digit0 : ",mean_feature1_digit0, " | variance_feature1_digit0 : ", variance_feature1_digit0)
    
    mean_feature2_digit0, variance_feature2_digit0 = mean_and_variance_feature2_calulation(train0_image_feature_matrix)
    print("\n mean_feature2_digit0 : ",mean_feature2_digit0, " | variance_feature2_digit0 : ", variance_feature2_digit0)

    
    train1_image_feature_matrix = task_1_feature_extraction(train1)
    #print(train1_image_feature_matrix[0])
    
    mean_feature1_digit1, variance_feature1_digit1 = mean_and_variance_feature1_calulation(train1_image_feature_matrix)
    print("\n mean_feature1_digit1 : ",mean_feature1_digit1, " | variance_feature1_digit1 : ", variance_feature1_digit1)
    
    mean_feature2_digit1, variance_feature2_digit1 = mean_and_variance_feature2_calulation(train1_image_feature_matrix)
    print("\n mean_feature2_digit1 : ",mean_feature2_digit1, " | variance_feature2_digit1 : ", variance_feature2_digit1)
    
    
    
    #label prediction for test 0 using test0 feature matrix
    test0_image_feature_matrix = task_1_feature_extraction(test0)
    #print(len(test0_image_feature_matrix))
    digit0_label_prediction = label_prediction(test0_image_feature_matrix, 
                                          mean_feature1_digit0, variance_feature1_digit0, 
                                          mean_feature2_digit0, variance_feature2_digit0,
                                          mean_feature1_digit1, variance_feature1_digit1, 
                                          mean_feature2_digit1, variance_feature2_digit1)
    #print(digit0_label_prediction)
    
    
    
    #label prediction for test1 using test1 feature matrix
    test1_image_feature_matrix = task_1_feature_extraction(test1)
    #print(len(test1_image_feature_matrix))
   
    digit1_label_prediction = label_prediction(test1_image_feature_matrix, 
                                          mean_feature1_digit0, variance_feature1_digit0, 
                                          mean_feature2_digit0, variance_feature2_digit0,
                                          mean_feature1_digit1, variance_feature1_digit1, 
                                          mean_feature2_digit1, variance_feature2_digit1)
    #print(digit1_label_prediction)
    
    
    digit0_test0_accuracy = digit0_accuracy_prediction(digit0_label_prediction)
    print("\n digit0_test0_accuracy : ",digit0_test0_accuracy)
    
    digit1_test1_accuracy = digit1_accuracy_prediction(digit1_label_prediction)
    print("\n digit1_test1_accuracy : ",digit1_test1_accuracy)
    
    

