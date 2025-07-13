// Author: Mattia Cozza

#ifndef YOLO_CONVERTER_HPP
#define YOLO_CONVERTER_HPP

#include <string>

void convertYoloToCsv(const std::string& labelFolder, const std::string& imageFolder, const std::string& outputCsv);

#endif
