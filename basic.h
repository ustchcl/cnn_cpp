#ifndef _BASIC_H_
#define _BASIC_H_

#include <vector>
#include <set>
#include <cassert>
#include "Size.hpp"
#include "Log.hpp"

#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <numeric>
//#include <boost/thread.hpp>
#include <functional>
// #include <boost/bind.hpp>
#include <boost/progress.hpp>


typedef std::vector<double> vector1d;
typedef std::vector<vector1d> vector2d;
typedef std::vector<vector2d> vector3d;
typedef std::vector<vector3d> vector4d;

static double ALPHA ;
static const double LAMBDA = 0;
static int recordInBatch = 0;
static int batchSize;
static int CPU_NUM = 4;

typedef double (*ProcessOne)(double);
typedef double (*ProcessTwo)(double, double);

double oneMinusValue(double value) {
    return 1 - value;
}

double sigmod(double value) {
    return 1.0 / (1 + exp(-value));
}




double plus(double value_1, double value_2) {
    return value_1 + value_2;
}

double multiply(double value_1, double value_2) {
    return value_1 * value_2;
}

double minus(double value_1, double value_2) {
    return value_1 - value_2;
}


// 网络层的类型：输入层、输出层、卷积层、采样层
enum LayerType {
    input,
    output,
    conv,
    samp
};


void matrixOp(
        const vector2d& ma, 
        const vector2d& mb,
        vector2d& result,
        ProcessOne funcA,
        ProcessOne funcB,
        ProcessTwo func
    ) {
    const int m = ma.size();
    int n = ma[0].size();
    
    
    assert(m == mb.size() && n == mb[0].size());
    assert(m == result.size() && n == result[0].size());
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double a = ma[i][j];
            if (funcA != NULL) a = (*funcA)(a);

            double b = mb[i][j];
            if (funcB != NULL) b = (*funcB)(b);

            assert(func != NULL);
            result[i][j] = func(a, b);
        }
    }
}

void matrixOp(const vector2d& ma, vector2d& result, ProcessOne func) {
    const int m = ma.size();
    int n = ma[0].size();
    
    assert(m == result.size() && n == result[0].size());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = (*func)(ma[i][j]);
        }
    }
}

void matrixOp(const vector2d& ma, vector2d& result, ProcessTwo func, double param) {
    const int m = ma.size();
    int n = ma[0].size();
    
    assert(m == result.size() && n == result[0].size());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = (*func)(ma[i][j], param);
        }
    }
}

void convnValid(const vector2d& matrix, 
                const vector2d& kernel,
                vector2d& result) {
    //kernel = rot180(kernel);
    int m = matrix.size();
    int n = matrix[0].size();
    const int km = kernel.size();
    const int kn = kernel[0].size();
    // 需要做卷积的列数
    int kns = n - kn + 1;
    // 需要做卷积的行数
    const int kms = m - km + 1;
    // 结果矩阵
    vector1d temp1d;
    temp1d.resize(kns);
    result.resize(kms, temp1d);

    for (int i = 0; i < kms; i++) {
        for (int j = 0; j < kns; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < km; ki++) {
                for (int kj = 0; kj < kn; kj++)
                    sum += matrix[i + ki][j + kj] * kernel[ki][kj];
            }
            result[i][j] = sum;
        }
    }
}


void convnFull(vector2d& matrix, const vector2d& kernel, vector2d& result) {
    int m = matrix.size();
    int n = matrix[0].size();
    const int km = kernel.size();
    const int kn = kernel[0].size();
    // 扩展矩阵
    vector1d temp1d;
    vector2d extendMatrix;
    temp1d.resize(n + 2 * (kn - 1));
    extendMatrix.resize(m + 2 * (km - 1), temp1d);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
        }
    }

    convnValid(extendMatrix, kernel, result);
}


void rot180(vector2d& matrix, vector2d& result) {
    vector1d temp1d;
    temp1d.resize(matrix[0].size());
    result.resize(matrix.size(), temp1d);
    
    int m = matrix.size();
    int n = matrix[0].size();
    
    // rot180
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n ; j++) {
            result[i][j] = matrix[n - 1 - i][n - 1 - j];
        }
    }

}

// 克罗克内积,对矩阵进行扩展
void kronecker(
        const vector2d& matrix,
        const Size& scale,
        vector2d& result
    ) {
    const int m = matrix.size();
    int n = matrix[0].size();
    
    vector1d temp1d;
    temp1d.resize(n * scale.y);
    result.resize(m * scale.x, temp1d);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
                for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
                    result[ki][kj] = matrix[i][j];
                }
            }
        }
    }
}


void scaleMatrix(
    const vector2d& matrix,
    const Size& scale,
    vector2d& result
    ) {
    
    int m = matrix.size();
    int n = matrix[0].size();
    const int sm = m / scale.x;
    const int sn = n / scale.y;
    vector1d temp1d;
    temp1d.resize(sn);
    result.resize(sm, temp1d);
    
    assert(sm * scale.x == m && sn * scale.y == n);

    const int size = scale.x * scale.y;
    for (int i = 0; i < sm; ++i) {
        for (int j = 0; j < sn; ++j) {
            double sum = 0.0;
            for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
                for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
                    sum += matrix[si][sj];
                }
            }
            result[i][j] = sum / size;
        }
    }
}

void randomPerm(int size, int batchSize, std::vector<int>& randPerm) {
    std::set<int> set;
    std::srand(time(0));
    while (set.size() < batchSize) {
        int value = (double)std::rand() / RAND_MAX * size;
        set.insert(value);
    }

    int i = 0;
    std::set<int>::iterator si;
    for (si = set.begin(); si != set.end(); ++si) {
        randPerm[i++] = (int)(*si);
    }
    set.clear();
}

/**
 * split string into substrings
 * str: input string
 * pattern: the split pattern
 */
void split(std::string str, std::string pattern, std::vector<std::string>& result)
{
    std::string::size_type pos;
    str += pattern;
    int size = str.size();
    for(int i = 0; i < size; ++i)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
}


/**
 * Cast value to type T
 */
template <typename T>
T parse(std::string value, T type)
{
    T valueT;
    try
    {
        std::stringstream ss(value);
        ss >> valueT;
    }
    catch (...)
    {
        std::cout << "Invalid Type!!" << std::endl;
    }
    return valueT;
}


void log(std::string s) {
    std::cout << s << std::endl;
}


auto randStart = std::chrono::system_clock::now();
auto randEnd = std::chrono::system_clock::now();

void timerStart() {
    randStart = std::chrono::system_clock::now();
}

void timerEnd() {
    randEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = randEnd - randStart;
    std::cout << "Use time " << diff.count() << " s" << std::endl;
}

void printSumMatrix(vector2d& m) {
	double sum = 0;
	for (int i = 0; i < m.size(); ++i) {
		for (int j = 0; j < m[i].size(); ++j) {
			sum += m[i][j];
		}

	}

	std::cout << sum << std::endl;
}


void printMatrix(vector1d& m, int X, int Y) {
	std::cout << std::endl;
	std::cout << std::endl;
	for (int i = 0; i < X; ++i) {
		for (int j = 0; j < Y; ++j) {
			std::cout << m[i * X + j] << " ";
		}
		std::cout << std::endl;
	}
}

void printMatrix(vector2d& m) {
	std::cout << std::endl;
	std::cout << std::endl;
	int X = m.size();
	int Y = m[0].size();
	for (int i = 0; i < X; ++i) {
		for (int j = 0; j < Y; ++j) {
			std::cout << m[i][j] << "\t";
		}
		std::cout << std::endl;
	}
}

#endif
