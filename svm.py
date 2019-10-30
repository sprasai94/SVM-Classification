import random
import math as m
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import MinMaxScaler


def Accuracy_Gamma_plot(results, kernel, Cs, gammas, grid_len):
    avg_score = results["mean_test_score"]
    if grid_len == 3:
        avg_score = np.concatenate((avg_score[0:4], avg_score[16:20], avg_score[32:36], avg_score[48:52])).reshape(len(Cs), len(gammas))
    else:
        avg_score = np.array(avg_score).reshape(len(Cs), len(gammas))

    plt.subplot(2, 2, kernels.index(kernel) + 1)

    for ind, i in enumerate(Cs):
        plt.plot(gammas, avg_score[ind], label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma (log scale)')
    plt.ylabel('Average accuracy')
    plt.xscale('log')
    plt.tight_layout()
    plt.title('Kernel: {}'.format(kernel))


# Reads the dataset, drops the first column that contains ID and returns the list of dataset
def read_datafile(filename):
    with open(filename, 'r') as ifl:
        array = [l.strip().split(',') for l in ifl]
        dataset = list(array)[:-1]
        [j.pop(0) for j in dataset]
        #shuflle the row of instances in dataset
        random.shuffle(dataset)
        features_num = len(dataset[0])
        for x in range(len(dataset)):
            for y in range(features_num):
                dataset[x][y] = float(dataset[x][y])

    return dataset


# perform grid search on the training sample
# performs grid search on ovo/ovr classifier based on parameter passed to it
# Also based on the class-weight type, the parameters of classifier is set to None or balanced
# plots the accuracy vs gamma for OvO classifier and unbalanced weight for each kernel
# returns the classifier model with best hyperparameters
def svm_grid_search(x_train, y_train, cv, classifier_type, class_type,  plot_count, kernel):
    c = [1, 10, 100, 1000]
    gamma = [0.0001, 0.001, 0.01, 0.1]
    degree = [2, 3, 4, 5]
    param_grid = {'estimator__C': c, 'estimator__gamma': gamma,'estimator__degree': degree}
    if classifier_type == 'ovo':
        clf = OneVsOneClassifier(svm.SVC(kernel=kernel, class_weight=class_type))
    elif classifier_type == 'ovr':
        clf = OneVsRestClassifier(svm.SVC(kernel=kernel, class_weight=class_type))
    clf_model = GridSearchCV(clf, param_grid=param_grid, cv=cv, iid=False)
    clf_model.fit(x_train, y_train)
    if classifier_type == 'ovo' and class_type is None and plot_count == 0:
        Accuracy_Gamma_plot(clf_model.cv_results_, kernel, c, gamma, len(param_grid))
    return clf_model


# Performs 5 fold cross validation on each kernel type of each classifier
# calls grid search to find the best fitted model
# predicts the label of the test set and find accuracy of classification
def SVM_Classifier_cross_validation(x, y, cv, classifier_type, class_type, kernel):
    total_sample = len(x)
    num_TestSample = int((1 / cv) * total_sample)
    t1 = 0
    t2 = num_TestSample
    accuracies = []
    training_time = []
    print('*****************************************************************************************************')
    print('Classifier:', classifier_type,'Class_weight:', class_type, 'Kernel:', kernel)
    for i in range(cv):
        # splitting tha dataset in test and training sample
        # spliting the dataset again to features x and label y
        x_test = x[t1:t2]
        y_test = y[t1:t2]
        x_train = np.concatenate((x[0:t1], x[t2:total_sample]), axis=0)
        y_train = np.concatenate((y[0:t1], y[t2:total_sample]), axis=0)

        t1 = t1 + num_TestSample
        t2 = t2 + num_TestSample
        start = time.time()
        tuned_model = svm_grid_search(x_train, y_train, cv, classifier_type, class_type, i, kernel=kernel)
        end = time.time()
        y_prediction = tuned_model.predict(x_test)

        accuracy = accuracy_score(y_prediction, y_test)
        accuracies.append(accuracy)
        training_time.append(end-start)
        print("Cv:", i, "Best parameters:", tuned_model.best_params_, 'Model accuracy:', tuned_model.best_score_,
              'Time:', end-start)
    print('Accuracies:', accuracies)
    average_accuracy = sum(accuracies) / len(accuracies)
    average_time = sum(training_time)/len(training_time)
    print('Average Test accuracy:', average_accuracy)
    print('Training time:', average_time)


if __name__== "__main__":
    # Reads the input file
    dataset = read_datafile("../Data/glass.data")
    # seperation of features and label
    x = [dataset[i][:-1] for i in range(len(dataset))]
    y = [dataset[i][-1] for i in range(len(dataset))]
    # Normalizing the features
    scalar = MinMaxScaler()
    x = scalar.fit_transform(x)

    cv = 5
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    classifier_type = ['ovo','ovr']
    class_type = [None, 'balanced']

    # For each classifier type, set class weight both balanced and unbalanced
    # For each classifier, use all of the four different kernels
    for i in range(len(classifier_type)):
        for j in range(len(class_type)):
            [SVM_Classifier_cross_validation(x, y, cv, classifier_type[i], class_type[j], kernel) for kernel in kernels]
    plt.show()
    print('End of program')