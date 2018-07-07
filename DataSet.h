/*
Author: Wang Wenzhu
Date: 2018/7/6
Description: A DataSet calss to manage INRIADATA
Usage: use the normalization folder as the root dir 
and pass it to the constructor to create a DateSet obj
then you can get the info of your dataset by calling obj.showInfo()
the img names in different subfolders are stored as opencv String vectors:
obj.tainPosList, trainNegList, testPosList, testNegList, trainHardList
for you to use
you MUST have your root dir organised like this:
©À©¤test
©¦  ©À©¤neg
©¦  ©¸©¤pos
©¸©¤train
©À©¤croppedNeg
©À©¤hard
©À©¤neg
©¸©¤pos
*/
#pragma once
#ifndef _DATASET_H
#define _DATASET_H

#include <iostream>
#include <opencv2\core\core.hpp>
#include <string>

using namespace cv;
using namespace std;

class DataSet {
public:
	//constructor
	DataSet(string dataDir= "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\");

	vector<String> trainPosImgList;
	vector<String> trainNegImgList;
	vector<String> testPosImgList;
	vector<String> testNegImgList;

	vector<String> trainHardList;
	
	void showInfo();

};
#endif
