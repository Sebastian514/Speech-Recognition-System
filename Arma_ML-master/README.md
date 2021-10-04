# Arma_ML (Armadillo-based Machine Learning library)
Author: Yuzhen Liu  
Started From 2019.3.22  
Light-weighted Statistic ML implementations in C++. Algorithms included are [linear regression](https://en.wikipedia.org/wiki/Linear_regression), [Logistic Regression Classifier](https://en.wikipedia.org/wiki/Logistic_regression), Sofrmax Classifier, [C4.5 Decision Tree](https://en.wikipedia.org/wiki/C4.5), [Random Forest](https://en.wikipedia.org/wiki/Random_forest), [GBDT](https://en.wikipedia.org/wiki/Gradient_boosting), FM, [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier), [SVM](https://en.wikipedia.org/wiki/Support-vector_machine).


## Installation
Requires C++ algebra library armadillo, installation of armadillo is as follows:  

install denpencies first (Tested for Ubuntu 16.04):  
    
    sudo apt-get install libopenblas-dev
    sudo apt-get install liblapack-dev
    sudo apt-get install libarpack2-dev
    sudo apt-get install libsuperlu-dev

download armadillo (armadillo-9.300.2 tested) as xxx.tar, cd and build it.   

	cd armadillo-9.300.2
	cmake .
	make
	sudo make install

clone and build Arma_ML directly in place

	git clone https://github.com/codestorm04/Arma_ML.git
	cd Arma_ML
	make

or build and install

	make install

examples/ are the usage demos of each modules, reference to [README.md](/examples/README.md)


## TODO:
1. Build strategies:   [Reference](https://www.cnblogs.com/Anker/p/3527677.html)
2. decision tree pruning
3. models: knn, LDA, PCA, MDS, k-means, FFM
4. params setting
5. normalization: L1 L2
6. optimazers other than SGD
7. model saver / loader
8. oprimizers, metrics, console visualization module


## Contrinutions, Issues and Starts are Welcomed :) !!! 
