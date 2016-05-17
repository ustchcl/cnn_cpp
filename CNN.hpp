#ifndef _CNN_HPP_
#define _CNN_HPP_

#include <algorithm>
#include <vector>
#include "Layer.hpp"
#include "Dataset.hpp"


class CNN {
private:

    
    std::vector<int> randPerm;

    // 网络的各层
    std::vector<Layer> layers;
    // 层数
    int layerNum;
    // 批大小
    
    // 除数操作符，对矩阵的每一个元素除以一个值
    ProcessOne divideBatchSize;
    // 乘数操作符，对矩阵的每一个元素乘以alpha值
	ProcessOne multiplyAlpha;
	// 乘数操作符，对矩阵的每一个元素乘以1-labmda*alpha值
	ProcessOne multiplyLambda;
    
    
    
    // 计算残差
    void setHiddenLayerErrors();
    // 采样残差
    void setSampErrors(Layer& layer, Layer& nextLayer);
    // 卷积层残差
    void setConvErrors(Layer& layer, Layer& nextLayer);

    // 输出层残差
    bool setOutLayerErrors(Record& record);
    void forward(Record& record);
    // 设置输入层的输出值
    void setInLayerOutput(Record& record);
    void setConvOutput(Layer& layer, Layer& lastLayer);
    void setSampOutput(Layer& layer, Layer& lastLayer);
    //void setSampOutputProcess(Layer& layer, Layer& lastLayer, int start, int end);


public:
    CNN(const int batchSize);
    void initProcessor();

    void train(Dataset& trainset, int repeat);
    bool train(Record& record);
    void predict(Dataset testset);
    double test(Dataset& testset);
    void predict(Dataset& testset, std::string fileName);
    bool backPropagation(Record& record);
    
    void setup(int batchSize);

    // 更新参数
    void updateParas();
    void updateBias(Layer& layer);
    void updateKernels(Layer& layer, Layer& lastLayer);
    

    void saveModel(std::string fileName);
    void loadModel(std::string fileName);
    
    static double funcDivideBatchSize(double value) { assert(batchSize != 0); return  value / batchSize; }
    static double funcMultiplyAlpha(double value) { return value * ALPHA; }
    static double funcMultiplyLambda(double value) { return value * (1 - LAMBDA * ALPHA); }

};



CNN::CNN(const int batchSizeParam) {
    Size inputSize(28, 28);
    Layer inputLayer = Layer::buildLayer(inputSize, input, 1);
    Size convnSize(5, 5);
    Size sampSize(2, 2);
    Size outputSize(1, 1);
    Layer convnLayer1 = Layer::buildLayer(convnSize, conv, 6);
    Layer sampLayer1 = Layer::buildLayer(sampSize, samp, 1);
    Layer convnLayer2 = Layer::buildLayer(convnSize, conv, 12);
    Layer sampLayer2 = Layer::buildLayer(sampSize, samp, 1);
    Layer outputLayer = Layer::buildLayer(outputSize, output, 10);
    
    layers.push_back(inputLayer);
    layers.push_back(convnLayer1);
    layers.push_back(sampLayer1);
    layers.push_back(convnLayer2);
    layers.push_back(sampLayer2);
    layers.push_back(outputLayer);

    std::cout << "layers build successfully..." << std::endl;

    layerNum = layers.size();
    batchSize = batchSizeParam;
    
    ALPHA = 0.85;
    
    setup(batchSize);
    
    randPerm.resize(batchSize);

    initProcessor();
}

void CNN::initProcessor() {
    std::cout << "Func in" << std::endl;

    divideBatchSize = &(CNN::funcDivideBatchSize);
    multiplyAlpha = &(CNN::funcMultiplyAlpha);
    multiplyLambda = &(CNN::funcMultiplyLambda);
    std::cout << "Func build successfully..." << std::endl;

}

void CNN::train(Dataset& trainset, int repeat) {
    for (int t = 0; t < repeat; t++) {
    	timerStart();
        int epochsNum = trainset.records.size() / batchSize;
        
        if (trainset.records.size() % batchSize != 0)
            epochsNum++;// 多抽取一次，即向上取整

        std::cout << "\n" << t << "th iter epochsNum: " << epochsNum <<  std::endl;
        int right = 0;
        int count = 0;
        boost::progress_display pd(epochsNum);

        for (int i = 0; i < epochsNum; i++) {
            // Generate an array of random numbers.
            std::vector<int>::iterator iv;

            randomPerm(trainset.size(), batchSize, randPerm);
            recordInBatch = 0;
            for (iv = randPerm.begin(); iv != randPerm.end(); ++iv) {
                int index = (int)(*iv);
                Record& record = trainset.records[index];
                bool isRight = train(record);
                if (isRight) ++right;
                ++count;
                ++recordInBatch;
            }

            // 跑完一个batch后更新权重
            updateParas();
        
            ++pd;

        }

        double p = 1.0 * right / count;
        if (t % 10 == 1 && p > 0.96) { // 动态调整准学习速率
            ALPHA = 0.001 + ALPHA * 0.9;
            std::cout << "set alpha = " << ALPHA << std::endl;
        }
        std::cout << "precision " << right << "/" << count << "=" << p << std::endl;
        timerEnd();
    }
}


bool CNN::train(Record& record) {
    forward(record);
    bool result = backPropagation(record);
    return result;
}

void CNN::predict(Dataset testset) {
			recordInBatch = 0;

			int isRight = 0;
			for (int i = 0; i < testset.records.size(); ++i) {
				Record& record = testset.records[i];
				forward(record);
				Layer& outputLayer = layers[layerNum - 1];

				int mapNum = outputLayer.getOutMapNum();
				vector1d out;
				out.resize(mapNum, 0);
				for (int m = 0; m < mapNum; m++) {
					vector2d& outmap = outputLayer.getMap(m);
					out[m] = outmap[0][0];
				}

				int label = (int)record.label;

				isRight++;
				for (int i = 0; i < out.size(); ++i) {
					if (out[i] > out[label]) {
						isRight --;
						break;
					}
				}
			}
			double p = 1.0 * isRight / testset.records.size();
			std::cout << "precision on testset " << isRight << "/" << testset.records.size() << "=" << p << std::endl;

	}


double CNN::test(Dataset& trainset) {
    Layer::prepareForNewBatch();
    
    int right = 0;
    for (int i = 0; i < trainset.records.size(); ++i) {
        Record record = trainset.records[i];
        forward(record);
        Layer& outputLayer = layers[layerNum - 1];
        int mapNum = outputLayer.getOutMapNum();
        
        vector1d out;
        out.resize(mapNum);
        for (int m = 0; m < mapNum; ++m) {
            vector2d& outmap = outputLayer.getMap(m);
            out[m] = outmap[0][0];
        }
        
        int index = (int) record.label;
        
        right++;
        for (int j = 0; j < mapNum; ++j) {
            if (out[j] > out[index]) {
                right--;
                break;
            }
        }
    }
    
    double p = 1.0 * right / trainset.records.size();
    return p;
}

// BP
bool CNN::backPropagation(Record& record) {
    bool result = setOutLayerErrors(record);
    setHiddenLayerErrors();
    return result;
}


// 更新参数

void CNN::updateParas() {
	//log("update params in...");
    for (int l = 1; l < layerNum; l++) {
        Layer& layer = layers[l];
        Layer& lastLayer = layers[l-1];
        
        if (layer.getType() == conv || layer.getType() == output) {
            updateKernels(layer, lastLayer);
            updateBias(layer);
        }
    }
    //log("update params out...");
}


void CNN::updateBias(Layer& layer) {
	//log("update bias in...");
/*
    std::function<void(Layer&, int, int)> process = [](Layer& layer, int start, int end) {
    	vector4d& errors = layer.getErrors();
    	int lengthI = errors.size();
    	int lengthJ = errors[0].size(); // outMapNum
    	int lengthK = errors[0][0].size();
    	int lengthL = errors[0][0][0].size();

		for (int j = start; j < end; j++) {
			double errorSum = 0;
			for (int i = 0; i < lengthI; ++i) {
				for (int k = 0; k < lengthK; ++k) {
					for (int l = 0; l < lengthL; ++l) {
						errorSum += errors[i][j][k][l];

						// 更新偏置
						double deltaBias = errorSum / batchSize;
						double bias = layer.getBias(j) + ALPHA * deltaBias;
						layer.setBias(j, bias);
					}
				}
			}
		}
    };


    boost::thread_group t_group;
    int mapNum = layer.getOutMapNum();
    int runCpu = CPU_NUM < mapNum ? CPU_NUM : 1;
    int fregLength = (mapNum + runCpu - 1) / runCpu;
    for (int cpu = 0; cpu < runCpu; cpu++) {
    	int start = cpu * fregLength;
        int tmp = (cpu + 1) * fregLength;
        int end = tmp <= mapNum ? tmp : mapNum;
        t_group.create_thread(boost::bind(process, layer, start, end));
        //std::cout << " thread " << cpu << ": started!" << std::endl;
    }
    t_group.join_all();
    //log("update bias out...");
*/


	vector4d& errors = layer.getErrors();
    int lengthI = errors.size();
    int lengthJ = errors[0].size(); // outMapNum
    int lengthK = errors[0][0].size();
    int lengthL = errors[0][0][0].size();

    for (int j = 0; j < lengthJ; j++) {
        double errorSum = 0;
        for (int i = 0; i < lengthI; ++i) 
            for (int k = 0; k < lengthK; ++k) 
                for (int l = 0; l < lengthL; ++l) 
                    errorSum += errors[i][j][k][l];

        // 更新偏置
        double deltaBias =  errorSum / batchSize;
        double bias = layer.getBias(j) + ALPHA * deltaBias;
        layer.setBias(j, bias);
    }

}

void CNN::updateKernels(Layer& layer, Layer& lastLayer) {

    int mapNum = layer.getOutMapNum();
    const int lastMapNum = lastLayer.getOutMapNum();

    for (int j = 0; j < mapNum; j++) {
        for (int i = 0; i < lastMapNum; i++) {
            // 对batch的每个记录delta求和
            vector2d deltaKernel;
            for (int r = 0; r < batchSize; r++) {
                vector2d& error = layer.getError(r, j);
                if (deltaKernel.size() == 0) {
                    convnValid(lastLayer.getMap(r, i), error, deltaKernel);
//                    Log::i("error");
//                    printMatrix(error);
//
//                    Log::i("map");
//                    printMatrix(lastLayer.getMap(r, i));
//
//                    Log::i("out");
//                    printMatrix(deltaKernel);
//
//                    exit(0);
                } else { // 累积求和
                    vector2d temp;
                    convnValid(lastLayer.getMap(r, i), error, temp);
                    
//                    printMatrix(lastLayer.getMap(r, i));

//                    printMatrix(error);
//
//                    printMatrix(temp);
//                    exit(0);

                    ProcessTwo plusFunc = &plus;
                    matrixOp(temp, deltaKernel, deltaKernel, NULL, NULL, plusFunc);
                }
            }

            // 除以batchSize
            matrixOp(deltaKernel, deltaKernel, divideBatchSize);
            // 更新卷积核
            vector2d& kernel = layer.getKernel(i, j);
            
            ProcessTwo plusFunc = &plus;
            vector2d temp2d;
            vector1d temp1d;
            temp1d.resize(deltaKernel[0].size());
            temp2d.resize(deltaKernel.size(), temp1d);
            
            matrixOp(kernel, deltaKernel, temp2d, multiplyLambda, multiplyAlpha, plusFunc);

            layer.setKernel(i, j, temp2d);
        }
    }
}

void CNN::setHiddenLayerErrors() {
    for (int l = layerNum - 2; l > 0; --l) {
        Layer& layer = layers[l];
        Layer& nextLayer = layers[l+1];
        switch (layer.getType()) {
        case samp:
            setSampErrors(layer, nextLayer);
            break;
        case conv:
            setConvErrors(layer, nextLayer);
            break;
        default: // 只有采样层和卷积层需要处理残差，输入层没有残差，输出层已经处理过
            break;
        }
    }
}

void CNN::setSampErrors(Layer& layer, Layer& nextLayer) {
    int mapNum = layer.getOutMapNum();
    const int nextMapNum = nextLayer.getOutMapNum();

    for (int i = 0; i < mapNum; i++) {
        vector2d sum; // 对每一个卷积进行求和
        for (int j = 0; j < nextMapNum; j++) {
            vector2d& nextError = nextLayer.getError(j);
            vector2d& kernel = nextLayer.getKernel(i, j);
            
            // 对卷积核进行180度旋转，然后进行full模式下得卷积
            vector2d kernelRot180;
            rot180(kernel, kernelRot180);
            
            if (sum.size() == 0) {
                convnFull(nextError, kernelRot180, sum);
            } else {
                vector2d convnResult;
                convnFull(nextError, kernelRot180, convnResult);
                ProcessTwo plusFunc = &plus;
                matrixOp(convnResult, sum, sum, NULL, NULL, plusFunc);
            }
            layer.setError(i, sum);
        }
    }
}


void CNN::setConvErrors(Layer& layer, Layer& nextLayer) {
    int mapNum = layer.getOutMapNum();
    for (int m = 0; m < mapNum; ++m) {
        Size& scale = nextLayer.getScaleSize();
        vector2d& nextError = nextLayer.getError(m);
        vector2d& map = layer.getMap(m);
        
        vector2d outMatrix;
        vector1d temp1d;
        temp1d.resize(map[0].size());
        outMatrix.resize(map.size(), temp1d);
        
        // 矩阵相乘，但对第二个矩阵的每个元素value进行1-value操作
        ProcessOne oneMinusFunc = &oneMinusValue;
        ProcessTwo mutiplyFunc = &multiply;
        matrixOp(map, map, outMatrix, NULL, oneMinusFunc, mutiplyFunc);
        
        vector2d kroneckerResult;
        kronecker(nextError, scale, kroneckerResult);
        
        matrixOp(outMatrix, kroneckerResult, outMatrix, NULL, NULL, mutiplyFunc);
        
//        Log::i("set convn errors");
//        printMatrix(nextError);
//        exit(0);

        layer.setError(m, outMatrix);
    }
}


// 设置输出层的残差值
bool CNN::setOutLayerErrors(Record& record) {

    Layer& outputLayer = layers[layerNum - 1];
    int mapNum = outputLayer.getOutMapNum();

    vector1d target;
    target.resize(mapNum, 0);
    
    vector1d outmaps;
    outmaps.resize(mapNum, 0);

    for (int m = 0; m < mapNum; m++) {
        vector2d& outmap = outputLayer.getMap(m);
        outmaps[m] = outmap[0][0];
    }

    int label = (int)record.label;

    target[label] = 1;

    for (int m = 0; m < mapNum; m++) {
        outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m]) * (target[m] - outmaps[m]));
    }
    
    bool isRight = true;

    for (int i = 0; i < outmaps.size(); ++i) {
        if (outmaps[i] > outmaps[label]) {
            isRight = false;
            break;
        }
    }
    return isRight;
}



void CNN::forward(Record& record) {
    // 设置输入层的map

    setInLayerOutput(record);


    for (int l = 1; l < layers.size(); ++l) {
        Layer& layer = layers[l];
        Layer& lastLayer = layers[l - 1];
        switch (layer.getType()) {
        case conv:// 计算卷积层的输出
            setConvOutput(layer, lastLayer);
            break;
        case samp:// 计算采样层的输出
            setSampOutput(layer, lastLayer);
            break;
        case output:// 计算输出层的输出,输出层是一个特殊的卷积层
            setConvOutput(layer, lastLayer);
            break;
        default:
            break;
        }
    }

}

// 根据记录值，设置输入层的输出值
void CNN::setInLayerOutput(Record& record) {
    Layer& inputLayer = layers[0];
    Size& mapSize = inputLayer.getMapSize();
    vector1d& attr = record.attrs;
    assert (attr.size() == mapSize.x * mapSize.y);

    for (int i = 0; i < mapSize.x; ++i) {
        for (int j = 0; j < mapSize.y; ++j) {
            // 将记录属性的一维向量弄成二维矩阵
            inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
        }
    }
}


void CNN::setConvOutput(Layer& layer, Layer& lastLayer) {
/*
	std::function<void(Layer&, Layer&, int, int)> process = [](Layer& layer, Layer& lastLayer, int start, int end) {
		int mapNum = layer.getOutMapNum();
		const int lastMapNum = lastLayer.getOutMapNum();
		ProcessTwo plusFunc = &plus;
		ProcessOne sigmodFunc = &sigmod;

		for (int j = start; j < end; ++j) {
			vector2d sum; // 对每一个输入map的卷积进行求和

			for (int i = 0; i < lastMapNum; i++) {
				vector2d& lastMap = lastLayer.getMap(i);
				vector2d& kernel = layer.getKernel(i, j);

				if (sum.size() == 0) {
					convnValid(lastMap, kernel, sum);
				} else {
					vector2d convnResult;
					convnValid(lastMap, kernel, convnResult);

					matrixOp(convnResult, sum, sum, NULL, NULL, plusFunc);
				}
			}

			const double bias = layer.getBias(j);

			matrixOp(sum, sum, plusFunc, bias);
			matrixOp(sum, sum, sigmodFunc);

			layer.setMapValue(j, sum);
		}
	};

    const int mapNum = layer.getOutMapNum();

    boost::thread_group t_group;

    timerStart();
    int runCpu = CPU_NUM < mapNum ? CPU_NUM : 1;
    int fregLength = (mapNum + runCpu - 1) / runCpu;
    for (int cpu = 0; cpu < runCpu; cpu++) {
    	int start = cpu * fregLength;
    	int tmp = (cpu + 1) * fregLength;
    	int end = tmp <= mapNum ? tmp : mapNum;

    	t_group.create_thread(boost::bind(process, layer, lastLayer, start, end));
    }

    t_group.join_all();
    timerEnd();
    exit(0);
*/
	int mapNum = layer.getOutMapNum();
	const int lastMapNum = lastLayer.getOutMapNum();
	ProcessTwo plusFunc = &plus;
	ProcessOne sigmodFunc = &sigmod;

    for (int j = 0; j < mapNum; ++j) {
        vector2d sum;// 对每一个输入map的卷积进行求和
        
        for (int i = 0; i < lastMapNum; i++) {
            vector2d& lastMap = lastLayer.getMap(i);
            vector2d& kernel = layer.getKernel(i, j);

            
            if (sum.size() == 0) {
                convnValid(lastMap, kernel, sum);
            } else {
                vector2d convnResult;
                convnValid(lastMap, kernel, convnResult);
                matrixOp(convnResult, sum, sum, NULL, NULL, plusFunc);
            }
        }
        
        const double bias = layer.getBias(j);
        
        matrixOp(sum, sum, plusFunc, bias);
        matrixOp(sum, sum, sigmodFunc);
        layer.setMapValue(j, sum);
    }
}

// 设置采样层的输出值，采样层是对卷积层的均值处理
void CNN::setSampOutput(Layer& layer, Layer& lastLayer) {
    int lastMapNum = lastLayer.getOutMapNum();

    for (int i = 0; i < lastMapNum; i++) {
        vector2d& lastMap = lastLayer.getMap(i);
        Size& scaleSize = layer.getScaleSize();

        // 按scaleSize区域进行均值处理
        vector2d sampMatrix;
        scaleMatrix(lastMap, scaleSize, sampMatrix);
        layer.setMapValue(i, sampMatrix);
    }
}

void CNN::setup(int batchSize) {
    log("setup in");
    Layer& inputLayer = layers[0];
    // 每一层都需要初始化输出map
    log("init outmap");
    inputLayer.initOutmaps(batchSize);
    for (int i = 1; i < layers.size(); ++i) {
        Layer& layer = layers[i];
        Layer& frontLayer = layers[i - 1];
        
        int frontMapNum = frontLayer.getOutMapNum();
        
        Size temp(0, 0);
        switch (layer.getType()) {
        case input:
            break;
        case conv:
            // 设置map的大小
            log("init conv");
            frontLayer.getMapSize().subtract(layer.getKernelSize(), 1, temp);
            layer.setMapSize(temp);
            
            // 初始化卷积核，共有frontMapNum*outMapNum个卷积核
            layer.initKernel(frontMapNum);
            
            // 初始化偏置，共有frontMapNum*outMapNum个偏置
            layer.initBias(frontMapNum);
            
            // batch的每个记录都要保持一份残差
            layer.initErros(batchSize);
            
            // 每一层都需要初始化输出map
            layer.initOutmaps(batchSize);
            log("end init conv");
            break;
        case samp:
            log("init samp");
            // 采样层的map数量与上一层相同
            layer.setOutMapNum(frontMapNum);
            
            // 采样层map的大小是上一层map的大小除以scale大小

            frontLayer.getMapSize().divide(layer.getScaleSize(), temp);
            layer.setMapSize(temp);
            
            // batch的每个记录都要保持一份残差
            layer.initErros(batchSize);
            
            // 每一层都需要初始化输出map
            layer.initOutmaps(batchSize);
            break;
        case output:
            log("init outmap");
            // 初始化权重（卷积核），输出层的卷积核大小为上一层的map大小
            layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());
            
            // 初始化偏置，共有frontMapNum*outMapNum个偏置
            layer.initBias(frontMapNum);
            
            // batch的每个记录都要保持一份残差
            layer.initErros(batchSize);
            
            // 每一层都需要初始化输出map
            layer.initOutmaps(batchSize);
            break;
        }
    }
}

#endif






