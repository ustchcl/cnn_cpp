#ifndef _SIZE_CPP_
#define _SIZE_CPP_


#include "MyException.hpp"
#include <iostream>

class Size {
public:
    int x;
    int y;
    Size();
    Size(int x, int y);
    void divide(Size& scaleSize, Size& result);
    void subtract(Size& size, int append, Size& result);
    
};



Size::Size() {
    this->x = 0;
    this->y = 0;
}

Size::Size(int x, int y) {
    this->x = x;
    this->y = y;
}
/**
* 整除scaleSize得到一个新的Size，要求this.x、this.
* y能分别被scaleSize.x、scaleSize.y整除
*/
void Size::divide(Size& scaleSize, Size& result) {
    int x = this->x / scaleSize.x;
    int y = this->y / scaleSize.y;
    
    try {
        if (x * scaleSize.x != this->x || y * scaleSize.y != this->y) {
            // throw MyException("不能整除 x=" + scaleSize.x + " y=" + scaleSize.y);
        }
    } catch(MyException& e) {
        std::cout << e.what() << std::endl;
    }
    result.x = x;
    result.y = y;

}

void Size::subtract(Size& size, int append, Size& result) {
    int x = this->x - size.x + append;
    int y = this->y - size.y + append;
    result.x = x;
    result.y = y;
}

#endif