#ifndef _LAYER_CPP_
#define _LAYER_CPP_

#include "basic.h"
#include "Size.hpp"
#include <string>
#include <cstdlib>

class Layer {
private:
    LayerType type;
    int outMapNum;
    int classNum;

    vector4d kernel;
    vector1d bias;
    vector4d outmaps;
    vector4d errors;
 /***   
    double[][][][] kernel;
    double[] bias;
    double[][][][] outmaps;
    double[][][][] errors;
     */

    //static int recordInBatch; // 记录当前训练的是batch的第几条记录
    Size mapSize; // map大小
    Size kernelSize; // 核大小
    Size scaleSize; // 采样

public:
    Layer();
    virtual ~Layer();
    static Layer buildLayer(Size& mapSize, LayerType type, int outMapNum);
    LayerType getType();
    
    int getOutMapNum();
    void setOutMapNum(int outMapNum);
    
    Size& getMapSize();
    void setMapSize(Size& mapSize);
    
    Size& getKernelSize();
    void setKernelSize(Size& size);

    Size& getScaleSize();
    
    static void prepareForNewBatch();
    static void prepareForNewRecord();
    
    // kernel
    void initKernel(int frontMapNum);
    void initOutputKerkel(int frontMapNum, Size& size);
    vector2d& getKernel(int i, int j);
    vector4d& getKernel();
    void setKernel(int i, int j, vector2d& kernel);
    
    // bias
    void initBias(int frontMapNum);
    double getBias(int mapNo);
    void setBias(int mapNo, double value);
    
    // outmaps
    void initOutmaps(int batchSize);
    void setMapValue(int mapNo, int mapX, int mapY, double value);
    vector2d& getMap(int index);
    vector2d& getMap(int recordId, int index);
    vector4d& getOutMaps();
    void setMapValue(int j, vector2d& map);
    
    // error
    void setErrorsValue(int mapNo, int x, int y, double value);
    vector2d& getError(int mapNo);
    vector2d& getError(int recordId, int mapNo);
    vector4d& getErrors();
    void initErros(int batchSize);
    void randomvector4d(vector4d& v);
    void setError(int i, vector2d& error);
    void setError(int mapNo, int mapX, int mapY, double value);
    
    
    
    
    
};

// Layer::Layer():mapSize(Size()), kernelSize(Size()), scaleSize(Size()) {
Layer::Layer() {   
    recordInBatch = 0;
}

Layer::~Layer() {
    
}

Layer Layer::buildLayer(Size& size, LayerType type, int outMapNum) {
    
    Layer layer;
    layer.classNum = -1;
    
    layer.type = type;
    layer.outMapNum = outMapNum;
    
    switch(type) {
    case input:
        layer.setMapSize(size);
        break;
    case conv:
        layer.kernelSize.x = size.x;
        layer.kernelSize.y = size.y;
        break;
    case samp:
        layer.scaleSize.x = size.x;
        layer.scaleSize.y = size.y;
        break;
    case output:
        layer.setMapSize(size);
        layer.classNum = outMapNum;
        layer.outMapNum = outMapNum;
        break;
    default:
        break;
    }

    
    
    return layer;
}

LayerType Layer::getType() {
    return this->type;
}

int Layer::getOutMapNum() {
    return outMapNum;
}

void Layer::setOutMapNum(int outMapNum) {
    this->outMapNum = outMapNum;
}

void Layer::setMapSize(Size& mapSize) {
    this->mapSize.x = mapSize.x;
    this->mapSize.y = mapSize.y;
}

Size& Layer::getMapSize() {
    return this->mapSize;
}

Size& Layer::getKernelSize() {
    return this->kernelSize;
}
void Layer::setKernelSize(Size& size) {
	this->kernelSize.x = size.x;
	this->kernelSize.y = size.y;
}

Size& Layer::getScaleSize() {
    return this->scaleSize;
}

void Layer::prepareForNewBatch() {
    recordInBatch = 0;
}

void Layer::prepareForNewRecord() {
    recordInBatch++;
}


/************************
 *****    kernel    *****
 ************************/

// 随机初始化卷积核
void Layer::initKernel(int frontMapNum) {
//	int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//	int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//	double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
    log("init kernel in");
    int kernelSizeX = this->kernelSize.x;
    int kernelSizeY = this->kernelSize.y;


    vector1d temp1d;
    temp1d.resize(kernelSizeY);
    vector2d temp2d;
    temp2d.resize(kernelSizeX, temp1d);
    vector3d temp3d;
    temp3d.resize(outMapNum, temp2d);

    this->kernel.resize(frontMapNum, temp3d);
    
    randomvector4d(this->kernel);
}

void Layer::initOutputKerkel(int frontMapNum, Size& size) {
    this->kernelSize.x = size.x;
    this->kernelSize.y = size.y;
    int kernelSizeX = this->kernelSize.x;
    int kernelSizeY = this->kernelSize.y;

    vector1d temp1d;
    temp1d.resize(kernelSizeY);
    vector2d temp2d;
    temp2d.resize(kernelSizeX, temp1d);
    vector3d temp3d;
    temp3d.resize(outMapNum, temp2d);
    this->kernel.resize(frontMapNum, temp3d);
        
    randomvector4d(this->kernel);

}

vector2d& Layer::getKernel(int i, int j) {
    return kernel[i][j];
}

vector4d& Layer::getKernel() {
    return kernel;
}

void Layer::setKernel(int i, int j, vector2d& kernel) {
    int lengthK = kernel.size();
    int lengthL = kernel[0].size();
    
    for (int k = 0; k < lengthK; ++k) {
        for (int l = 0; l < lengthL; ++l) {
            this->kernel[i][j][k][l] = kernel[k][l];
        }
    }
}


/************************
 *****    bias      *****
 ************************/
void Layer::initBias(int frontMapNum) {
    this->bias.resize(this->outMapNum, 0);
}

double Layer::getBias(int mapNo) {
    return bias[mapNo];
}

void Layer::setBias(int mapNo, double value) {
    bias[mapNo] = value;
}

/************************
 *****   outmaps    *****
 ************************/
void Layer::initOutmaps(int batchSize) {
    int mapSizeX = this->mapSize.x;
    int mapSizeY = this->mapSize.y;
    
    vector1d temp1d;
    temp1d.resize(mapSizeY, 0);
    vector2d temp2d;
    temp2d.resize(mapSizeX, temp1d);
    vector3d temp3d;
    temp3d.resize(outMapNum, temp2d);

    this->outmaps.resize(batchSize, temp3d);
}

void Layer::setMapValue(int mapNo, int mapX, int mapY, double value) {
    outmaps[recordInBatch][mapNo][mapX][mapY] = value;
}

vector2d& Layer::getMap(int index) {
    return outmaps[recordInBatch][index];
}

vector2d& Layer::getMap(int recordId, int index) {
    return outmaps[recordId][index];
}

vector4d& Layer::getOutMaps() {
    return outmaps;
}

void Layer::setMapValue(int mapNo, vector2d& map) {
    int lengthK = map.size();
    int lengthL = map[0].size();
    
    for (int k = 0; k < lengthK; ++k) {
        for (int l = 0; l < lengthL; ++l) {
            this->outmaps[recordInBatch][mapNo][k][l] = map[k][l];
        }
    }
}



/************************
 *****    error     *****
 ************************/
void Layer::setErrorsValue(int mapNo, int x, int y, double value) {
    outmaps[recordInBatch][mapNo][x][y] = value;
}

vector2d& Layer::getError(int mapNo) {
    return errors[recordInBatch][mapNo];
}

vector2d& Layer::getError(int recordId, int mapNo) {
    return errors[recordId][mapNo];
}

vector4d& Layer::getErrors() {
    return errors;
}

void Layer::initErros(int batchSize) {
    int mapSizeX = this->mapSize.x;
    int mapSizeY = this->mapSize.y;
    
    vector1d temp1d;
    temp1d.resize(mapSizeY);
    vector2d temp2d;
    temp2d.resize(mapSizeX, temp1d);
    vector3d temp3d;
    temp3d.resize(outMapNum, temp2d);
    errors.resize(batchSize, temp3d);
}

void Layer::setError(int mapNo, vector2d& error) {
    int lengthK = error.size();
    int lengthL = error[0].size();
    
    for (int k = 0; k < lengthK; ++k) {
        for (int l = 0; l < lengthL; ++l) {
            this->errors[recordInBatch][mapNo][k][l] = error[k][l];
        }
    }
}

void Layer::setError(int mapNo, int mapX, int mapY, double value) {
    errors[recordInBatch][mapNo][mapX][mapY] = value;
}



void Layer::randomvector4d(vector4d& v) {
    int lengthI = v.size();
    int lengthJ = v[0].size();
    int lengthK = v[0][0].size();
    int lengthL = v[0][0][0].size();
    
    std::srand(time(0));
    
    for (int i = 0; i < lengthI; ++i)
        for (int j = 0; j < lengthJ; ++j)
            for (int k = 0; k < lengthK; ++k)
                for (int l = 0; l < lengthL; ++l) {
                    // 随机值在[-0.05,0.05)之间，让权重初始化值较小，有利于于避免过拟合
                	double temp = (double)std::rand() - RAND_MAX / 2;
                    kernel[i][j][k][l] = temp / RAND_MAX / 20;
                }
}
    

#endif
