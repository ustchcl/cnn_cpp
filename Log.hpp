
#ifndef _LOG_HPP_
#define _LOG_HPP_

#include <iostream>
#include <string>

class Log {
public:
	Log(){};
	static void i(std::string s);
	static void i(double d);
};


void Log::i(std::string s) {
	std::cout << s << std::endl;
}

void Log::i(double d) {
	std::cout << d << std::endl;
}

#endif

