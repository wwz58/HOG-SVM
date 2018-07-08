/*
Author: Wang Wenzhu
Date: 2018/7/6
Description: Detector calss
Usage: 
1.train and test
Detector("train") obj;
obj.firstTrain()
obj.testOnTestPosDataSet()...

obj.retrain()
obj.testOnTestPosDataSet()...

2.Detector("test") load a trained model
obj.testOnTestPosDataSet()...

*/
#pragma once
#ifndef _DETECCTOR_H
#define _DETECCTOR_H
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>

#include "DataSet.h"

using namespace cv;
using namespace std;

class Detector {
public:
	//constructor
	Detector(string trainOrTest,
		string dataPath = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\",
		string hogTxt = "hogDetectorAllHard.txt");

	void firstTrain();

	void reTrain();

	void testOnTestPosDataSet();
	void testOnTestNegDataSet();
	void testPic();
	void testVideo();

private:
	void genHogArg();//pay attention to hard example or not 
	vector<float> loadHogTxt(string hogTxt);//use in hog.setSVMDetector(*)
	

	static Ptr<ml::SVM>svm;
	static HOGDescriptor hog;
	//Ptr<ml::SVM>svm;
	//HOGDescriptor hog;
	DataSet data;

	vector<float> myDetector;

};
#endif
