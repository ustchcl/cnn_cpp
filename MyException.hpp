#include <string>

class MyException {
public:
    std::string infoMsg;

    MyException();
    MyException(std::string infoMsg);
    ~MyException();
    
    std::string what();
};

MyException::MyException(std::string infoMsg) {
    this->infoMsg = infoMsg;
}

std::string MyException::what() {
    return this->infoMsg;
}

