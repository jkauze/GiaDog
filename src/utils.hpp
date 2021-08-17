/*
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 2021/08/13

	Utilities.
*/
#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

/* Imprime un mensaje de error y finaliza la ejecucion. */
void error(std::string msg) {
	std::cerr << "\e[1;31mError. \e[0m" << msg << "\n";
	exit(1);
}

/* Divide un string. Similar al split de Python. */
std::vector<std::string> split(const std::string &s, char delim=' ') {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) 
  {
    elems.push_back(item);
  }
  return elems;
}