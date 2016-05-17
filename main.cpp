#include <stdio.h>
#include "CNN.hpp"


int main(int argc, char **argv)
{
    
    using namespace std;
    
    cout << "Start..." << endl;

    CNN cnn(50);
    cout << "cnn build successfully..." << endl;
    string fileName("dataset/train.format");
    Dataset trainset(784);
    //trainset.loadFile(fileName, ",");
     trainset.readTestFile("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");
    // trainset.readTestFile("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");
    cout << "trainset build successfully..." << endl;
    
    cnn.train(trainset, 10);
    trainset.clear();
    
    Dataset testset(-1);
    testset.readTestFile("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");
    
    cnn.predict(testset);
    
	printf("hello world\n");
}


