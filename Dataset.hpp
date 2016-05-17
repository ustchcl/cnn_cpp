#ifndef _DATASET_HPP_
#define _DATASET_HPP_

#include "basic.h"
#include <fstream>

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

class Record {
public:
    // 存储数据
    std::vector<double> attrs;
    double label;

	Record(std::vector<double>& attrs, double lable);
    Record(std::vector<double>& data, int labelIndex, double& maxLabel); 
};


Record::Record(std::vector<double>& attrsParam, double labelParam) {
    this->attrs.resize(attrsParam.size());
    for (int i = 0; i < attrs.size(); ++i) {
        attrs[i] = attrsParam[i];
    }
    this->label = labelParam;
}

Record::Record(std::vector<double>& data, int labelIndex, double& maxLabel) {
    // 无便签
    if (labelIndex == -1) {
        this->attrs.resize(data.size());
        for (int i = 0; i < attrs.size(); ++i) {
            attrs[i] = data[i];
        }
    } else {
        this->label = data[labelIndex];
        if (label > maxLabel) {
            maxLabel = label;
        }
        
        if (labelIndex == 0) {
            this->attrs.resize(data.size() - 1);
            for (int i = 0; i < attrs.size() - 1; ++i) {
                attrs[i] = data[i + 1];
            }
        } else {
            this->attrs.resize(data.size() - 1);
            for (int i = 0; i < attrs.size() - 1; ++i) {
                attrs[i] = data[i];
            }
        }
    }
}




class Dataset {
public:
    std::vector<Record> records;
    int labelIndex;
    
    double maxLabel;
    Dataset(int classIndex) : labelIndex(classIndex) {}
    
    int size() { return records.size(); }
    int getLableIndex() { return labelIndex; }
    void append(Record record) { records.push_back(record);}
    void loadFile(std::string filePath, std::string tag);
    void clear() { records.clear(); }
    void readTestFile(std::string imgFileName, std::string labelFileName);
    
};

void Dataset::loadFile(std::string filePath, std::string tag) {
    
    std::cout << "load file in ..." << std::endl;
    boost::progress_display pd(12000);

    std::ifstream ifs(filePath.c_str());
    assert(ifs.is_open());
    
    std::string line;
    std::vector<std::string> datas;
    
    std::vector<double> data;
    int count = 0;
    while (std::getline(ifs, line)) {
        datas.clear();
        split(line, tag, datas);
        
        data.resize(datas.size(), 0);
        
        for (int i = 0; i < datas.size(); ++i) {
            data[i] = std::atof(datas[i].c_str());            
        }
        Record record(data, labelIndex, maxLabel);
        records.push_back(record);

        ++pd;
    }

    log ("end load " + filePath);
}




void Dataset::readTestFile(std::string fileName, std::string labelFileName) {
	std::ifstream ifsImg(fileName.c_str(),std::ios::in | std::ios::binary);
	std::ifstream ifsLabel(labelFileName.c_str(),std::ios::in | std::ios::binary);

	assert(ifsImg.is_open());
	assert(ifsLabel.is_open());
	int magic_number = 0;
	int magic_number_label = 0;
		int number_of_images = 0;
		int number_of_images_label= 0;
		int rows = 0;
		int cols = 0;

		ifsImg.read((char*)&magic_number,sizeof(magic_number));
		magic_number= reverseInt(magic_number);
		ifsImg.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= reverseInt(number_of_images);
		ifsImg.read((char*)&rows,sizeof(rows));
		rows= reverseInt(rows);
		ifsImg.read((char*)&cols,sizeof(cols));
		cols= reverseInt(cols);



		// for label
		ifsLabel.read((char*)&magic_number_label,sizeof(magic_number_label));
		magic_number_label= reverseInt(magic_number_label);
		ifsLabel.read((char*)&number_of_images_label,sizeof(number_of_images_label));
		number_of_images_label= reverseInt(number_of_images_label);
		std::cout << magic_number << " " << number_of_images << " " << rows << " " << cols << std::endl;

		std::vector<double> data;
		for (int i = 0; i < number_of_images; i++) {
			data.resize(rows * cols, 0);

			for(int row = 0; row < rows; row++){
				for(int col = 0; col < cols; col++){
					unsigned char temp = 0;
					ifsImg.read((char*)&temp,sizeof(temp));

					// 二值化
//					if (temp > 0) {
//						temp = 1;
//					}
					data[rows*row+col] = (double)temp;
				}
			}
			unsigned char tempLabel = 0;
			ifsLabel.read((char*)&tempLabel,sizeof(tempLabel));
//			printMatrix(data, 28, 28);


			double label = (double)tempLabel;
//			Log::i(label);
//			exit(0);
			Record record(data, label);
			records.push_back(record);
		}
		ifsImg.close();
		ifsLabel.close();
}







#endif
