#include "Detector.h"
using namespace cv;
using namespace std;

int main(int argc,char* argv) {
	

	//1.train and test
		Detector obj("train");
		//obj.firstTrain();
		//obj.testOnTestPosDataSet();
		
		obj.reTrain();
		//obj.testOnTestPosDataSet();

	//2.Detector("test") load a trained model
		//Detector obj("test");

		//obj.testOnTestPosDataSet();
		//obj.testOnTestNegDataSet();
		obj.testPic();
		//obj.testVideo();
		//DataSet data= DataSet();
		//data.showInfo();

		getchar();
}