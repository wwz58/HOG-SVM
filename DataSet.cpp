#include "DataSet.h"



DataSet::DataSet(string dataDir)
{
	string trainPosDir = dataDir + "train\\pos\\";
	string trainNegDir = dataDir + "train\\croppedNeg\\";
	string trainHardNegDir = dataDir + "train\\hard\\";

	string testPosDir = dataDir + "test\\pos\\";
	string testNegDir = dataDir + "test\\neg\\";

	String patternJPG("*.jpg");
	String patternPNG("*.png");

	glob(String(trainPosDir)+patternPNG,trainPosImgList,false);
	glob(String(trainNegDir)+patternJPG,trainNegImgList,false);
	glob(String(testPosDir)+patternPNG,testPosImgList ,false);
	glob(String(testNegDir)+patternJPG, testNegImgList, false);

	glob(String(trainHardNegDir)+patternJPG, trainHardList,false);
	
}

void DataSet::showInfo()
{
	cout << "your dataset info:" << endl;
	cout << "Number of pos data used for train:" << trainPosImgList.size() << endl;
	cout << "Number of neg data used for train:" << trainNegImgList.size() << endl;
	cout << "Number of pos data used for test:" << testPosImgList.size() << endl;
	cout << "Number of neg data used for test:" << testNegImgList.size() << endl;
	cout << "Number of hard data used for train:" << trainHardList.size() << endl;
	if (trainHardList.size() == 0)
		cout << "No hard example\n, you can start train a svm detector\n then test it to generate some hard example for retraining" << endl;
	else
		cout << "you already have hard example,\n if you haven't done a retraining yet,\n you may start retraining now" << endl;

}