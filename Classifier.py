from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
from sklearn.naive_bayes import GaussianNB
import numpy as np
#import our data
class classifier():
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.X_train,\
        self.X_test,\
        self.y_train,\
        self.y_test = ts(self.X, self.y, test_size=0.1)
    def SVM(self):

# iris = datasets.load_iris()
# X = np.loadtxt("features.txt")
# y = np.loadtxt("label.txt")
# print(X)
# print(y)
#split the data to  7:3
        

        # select different type of kernel function and compare the score

        # kernel = 'rbf'
        clf_rbf = svm.SVC(kernel='rbf')
        clf_rbf.fit(self.X_train, self.y_train)
        score_rbf = clf_rbf.score(self.X_test,self.y_test)
        print("The score of rbf is : %f"%score_rbf)

        # kernel = 'linear'
        clf_linear = svm.SVC(kernel='linear')
        clf_linear.fit(self.X_train,self.y_train)
        score_linear = clf_linear.score(self.X_test,self.y_test)
        print("The score of linear is : %f"%score_linear)

        # kernel = 'poly'
        clf_poly = svm.SVC(kernel='poly')
        clf_poly.fit(self.X_train,self.y_train)
        score_poly = clf_poly.score(self.X_test,self.y_test)
        print("The score of poly is : %f"%score_poly)
    def Naive_Bayes(self):
        clf = GaussianNB()
        clf = clf.fit(self.X , self.y)
        score = clf.score( self.X_test, self.y_test)
        print("Naive Bayes %f" %score)
    def Fisher_LDA(self):
        
    def DecisionTree(self):
