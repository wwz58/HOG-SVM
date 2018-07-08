#include "Detector.h"

Ptr<ml::SVM> Detector::svm = ml::SVM::create();
HOGDescriptor Detector::hog = HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);


Detector::Detector(string trainOrTest, string dataPath, string hogTxt):data(dataPath)
{
	//svm = ml::SVM::create();
	if (trainOrTest=="train") {
		svm->setType(ml::SVM::C_SVC);
		svm->setKernel(ml::SVM::LINEAR);
		svm->setGamma(1);
		svm->setC(0.01);
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代 
		svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

		cout << "可以开始训练SVM分类器了..." << endl;
	}
	else if(trainOrTest == "test"){
		//load svm xml
		svm = ml::SVM::load("svmLinearAllHard.xml");
		//load hog txt and set hog
		myDetector = loadHogTxt(hogTxt);
		hog.setSVMDetector(myDetector);//设置检测器参数为自己训练的SVM参数
		cout << "可以开始检测了..." << endl;
	}
}


void Detector::firstTrain()
{
	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

	//正样本
	for (int num = 0; num < data.trainPosImgList.size(); num++) {

		cout << "处理正样本..."<<num+1<< endl;
		Mat src = imread(data.trainPosImgList[num]);//读取图片
		src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子

		if (0 == num)
		{
			int DescriptorDim = descriptors.size();//HOG描述子的维数
			cout << "描述子维数：" << DescriptorDim << endl;
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(data.trainPosImgList.size() + data.trainNegImgList.size(), DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(sampleFeatureMat.rows, 1, CV_32SC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素

		sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
	}

	//负样本
	for (int num = 0; num < data.trainNegImgList.size(); num++) {

		cout << "处理负样本..."<< num+1 << endl;
		Mat src = imread(data.trainNegImgList[num]);//读取图片
		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num + data.trainPosImgList.size(), i) = descriptors[i];

		sampleLabelMat.at<float>(num + data.trainPosImgList.size(), 0) = -1;//正样本类别为1，有人
	}

	//train
	svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);
	cout << "训练完成" << endl;

	//save xml and hog.setSVM then save txt then test using this trained svm+hog
	svm->save("test.xml");//将训练好的SVM模型保存为xml文件
	cout << "模型已保存到test.xml" << endl;

	//save hog txt and set hog
	genHogArg();
	cout << "模型已保存到test.xml" << endl;
	hog.setSVMDetector(myDetector);
	cout << "hog已使用first train 的SVM设置\n 可以开始检测了..." << endl;

}


void Detector::reTrain()
{
	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

	//正样本
	for (int num = 0; num < data.trainPosImgList.size(); num++) {

		cout << "处理正样本..."<<num+1 << endl;
		Mat src = imread(data.trainPosImgList[num]);//读取图片
		src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors,Size(8,8));//计算HOG描述子

		if (0 == num)
		{
			int DescriptorDim = descriptors.size();//HOG描述子的维数
			cout << "描述子维数：" << DescriptorDim << endl;
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(data.trainPosImgList.size() + data.trainNegImgList.size()+data.trainHardList.size(), DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(sampleFeatureMat.rows, 1, CV_32SC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素

		sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
	}

	//负样本
	for (int num = 0; num < data.trainNegImgList.size(); num++) {
		cout << "处理负样本..."<<num+1 << endl;
		Mat src = imread(data.trainNegImgList[num]);//读取图片
		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num + data.trainPosImgList.size(), i) = descriptors[i];

		sampleLabelMat.at<float>(num + data.trainPosImgList.size(), 0) = -1;//正样本类别为1，有人
	}
	
	//难样本
	for (int num = 0; num < data.trainHardList.size(); num++) {
		cout << "处理难样本..." <<num+1<< endl;
		Mat src = imread(data.trainHardList[num]);//读取图片
		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num + data.trainPosImgList.size()+data.trainNegImgList.size(), i) = descriptors[i];

		sampleLabelMat.at<float>(num + data.trainPosImgList.size() + data.trainNegImgList.size(), 0) = -1;//正样本类别为1，有人
	}

	//train
	svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);
	cout << "训练完成" << endl;

	//save xml and hog.setSVM then save txt then test using this trained svm+hog
	svm->save("test2.xml");//将训练好的SVM模型保存为xml文件
	cout << "模型已保存到test2.xml" << endl;

	//save hog txt and set hog
	genHogArg();
	cout << "模型已保存到test2.xml" << endl;
	hog.setSVMDetector(myDetector);
	cout << "hog已使用re-train 的SVM设置\n 可以开始检测了..." << endl;

}

void Detector::testOnTestPosDataSet()
{
	int rightNum = 0;

	for (int i = 0; i < data.testPosImgList.size(); i++) {
		cout << "开始检测..."<<i << endl;
		Mat src = imread(data.testPosImgList[i]);
		src = src(Rect(3, 3, 64, 128));
		vector<float> descriptor;
		hog.compute(src, descriptor, Size(8, 8));
		Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//测试样本的特征向量矩阵
		for (int j = 0; j < descriptor.size(); j++)
			testFeatureMat.at<float>(0, j) = descriptor[j];
		double result = svm->predict(testFeatureMat);//返回类标
		if (result >= 1.0)
			rightNum++;
		cout << "结果：" << result << endl;
	}
	cout << "处理完毕！" << endl;
	float precision = float(rightNum) / data.testPosImgList.size();
	cout << "TP=" << rightNum << endl;
	cout << "正样本测试集上的准确率：" << precision << endl;
}

void Detector::testOnTestNegDataSet()
{
	int rightNum = 0;

	for (int i = 0; i < data.testNegImgList.size(); i++) {
		cout << "开始检测..." << i << endl;
		Mat src = imread(data.testNegImgList[i]);
		vector<float> descriptor;
		hog.compute(src, descriptor, Size(8, 8));
		Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//测试样本的特征向量矩阵
		for (int j = 0; j < descriptor.size(); j++)
			testFeatureMat.at<float>(0, j) = descriptor[j];
		double result = svm->predict(testFeatureMat);//返回类标
		if (result <= -1.0)
			rightNum++;
		cout << "结果：" << result << endl;
	}
	cout << "处理完毕！" << endl;
	float precision = float(rightNum) / data.testNegImgList.size();
	cout << "TN=" << rightNum << endl;
	cout << "负样本测试集上的准确率：" << precision << endl;
}

void Detector::testPic() {
	Mat srcc = imread("timgRD981YXB.jpg");
	Mat src = srcc.clone();
	vector<Rect> found, found_filtered;//矩形框数组
	cout << "进行多尺度HOG人体检测" << endl;
	hog.detectMultiScale(src, found, 0, Size(8, 8), Size(16, 16), 1.05, 2);//对图片进行多尺度行人检测
	cout << "找到的矩形框个数：" << found.size() << endl;

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for (int i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.15);
		r.width = cvRound(r.width*0.7);
		r.y += cvRound(r.height*0.1);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(255, 0, 0), 3);
	}

	//imwrite("ImgProcessed.jpg", src);
	namedWindow("src", 0);
	imshow("src", src);

	waitKey(10);
}

void Detector::testVideo()
{
	VideoCapture cap;
	cap.open("C:\\Users\\20203\\Desktop\\multi_people.mp4");
	if (!cap.isOpened()) {
		cout << "打开文件失败\n";
		return;
	}
	Mat frame;

	for (;; )
	{
		cap >> frame;
		if (frame.empty())
			break;
		vector<Rect> found, found_filtered;//矩形框数组
										   //cout << "进行多尺度HOG人体检测" << endl;
		hog.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//对图片进行多尺度行人检测
																				   //cout << "找到的矩形框个数：" << found.size() << endl;

																				   //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
		for (int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j = 0;
			for (; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
		for (int i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.15);
			r.width = cvRound(r.width*0.7);
			r.y += cvRound(r.height*0.1);
			r.height = cvRound(r.height*0.8);
			rectangle(frame, r.tl(), r.br(), Scalar(255, 0, 0), 3);
		}

		//imwrite("ImgProcessed.jpg", frame);
		namedWindow("src", 0);
		imshow("src", frame);
		waitKey(10);
		cout << "处理帧" << endl;
	}
	cap.release();
}

void Detector::genHogArg()
{
	//get support vectors
	int DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数
	Mat supportVector = svm->getSupportVectors();
	int supportVectorNum = supportVector.rows;//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	//get rho
	vector<float> svm_alpha;
	vector<float> svm_svidx;
	float svm_rho;
	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);

	//get resultMat
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha行向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵，每行一个支持向量
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果 行向量

	supportVectorMat = supportVector;

	//--将alpha向量的数据复制到alphaMat中
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = svm_alpha[i];
	}

	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm_rho);
	cout << "检测子维数：" << myDetector.size() << endl;

	//保存检测子参数到文件
	ofstream fout("test.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}
	cout << "保存参数到txt成功！" << endl;
}

vector<float> Detector::loadHogTxt(string hogTxt)
{
	ifstream fin_detector(hogTxt);//打开自己训练的SVM检测器文件
	vector<float>myDetector;

	while (!fin_detector.eof())
	{
		float temp;
		fin_detector >> temp;
		myDetector.push_back(temp);//放入检测器数组
	}
	cout << "检测子维数：" << myDetector.size() << endl;

	return myDetector;
}
