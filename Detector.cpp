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
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ���� 
		svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

		cout << "���Կ�ʼѵ��SVM��������..." << endl;
	}
	else if(trainOrTest == "test"){
		//load svm xml
		svm = ml::SVM::load("svmLinearAllHard.xml");
		//load hog txt and set hog
		myDetector = loadHogTxt(hogTxt);
		hog.setSVMDetector(myDetector);//���ü��������Ϊ�Լ�ѵ����SVM����
		cout << "���Կ�ʼ�����..." << endl;
	}
}


void Detector::firstTrain()
{
	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����

	//������
	for (int num = 0; num < data.trainPosImgList.size(); num++) {

		cout << "����������..."<<num+1<< endl;
		Mat src = imread(data.trainPosImgList[num]);//��ȡͼƬ
		src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG������

		if (0 == num)
		{
			int DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
			cout << "������ά����" << DescriptorDim << endl;
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
			sampleFeatureMat = Mat::zeros(data.trainPosImgList.size() + data.trainNegImgList.size(), DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(sampleFeatureMat.rows, 1, CV_32SC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��

		sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
	}

	//������
	for (int num = 0; num < data.trainNegImgList.size(); num++) {

		cout << "��������..."<< num+1 << endl;
		Mat src = imread(data.trainNegImgList[num]);//��ȡͼƬ
		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG������
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num + data.trainPosImgList.size(), i) = descriptors[i];

		sampleLabelMat.at<float>(num + data.trainPosImgList.size(), 0) = -1;//���������Ϊ1������
	}

	//train
	svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);
	cout << "ѵ�����" << endl;

	//save xml and hog.setSVM then save txt then test using this trained svm+hog
	svm->save("test.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	cout << "ģ���ѱ��浽test.xml" << endl;

	//save hog txt and set hog
	genHogArg();
	cout << "ģ���ѱ��浽test.xml" << endl;
	hog.setSVMDetector(myDetector);
	cout << "hog��ʹ��first train ��SVM����\n ���Կ�ʼ�����..." << endl;

}


void Detector::reTrain()
{
	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����

	//������
	for (int num = 0; num < data.trainPosImgList.size(); num++) {

		cout << "����������..."<<num+1 << endl;
		Mat src = imread(data.trainPosImgList[num]);//��ȡͼƬ
		src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors,Size(8,8));//����HOG������

		if (0 == num)
		{
			int DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
			cout << "������ά����" << DescriptorDim << endl;
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
			sampleFeatureMat = Mat::zeros(data.trainPosImgList.size() + data.trainNegImgList.size()+data.trainHardList.size(), DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(sampleFeatureMat.rows, 1, CV_32SC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��

		sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
	}

	//������
	for (int num = 0; num < data.trainNegImgList.size(); num++) {
		cout << "��������..."<<num+1 << endl;
		Mat src = imread(data.trainNegImgList[num]);//��ȡͼƬ
		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG������
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num + data.trainPosImgList.size(), i) = descriptors[i];

		sampleLabelMat.at<float>(num + data.trainPosImgList.size(), 0) = -1;//���������Ϊ1������
	}
	
	//������
	for (int num = 0; num < data.trainHardList.size(); num++) {
		cout << "����������..." <<num+1<< endl;
		Mat src = imread(data.trainHardList[num]);//��ȡͼƬ
		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG������
		for (int i = 0; i < descriptors.size(); i++)
			sampleFeatureMat.at<float>(num + data.trainPosImgList.size()+data.trainNegImgList.size(), i) = descriptors[i];

		sampleLabelMat.at<float>(num + data.trainPosImgList.size() + data.trainNegImgList.size(), 0) = -1;//���������Ϊ1������
	}

	//train
	svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);
	cout << "ѵ�����" << endl;

	//save xml and hog.setSVM then save txt then test using this trained svm+hog
	svm->save("test2.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	cout << "ģ���ѱ��浽test2.xml" << endl;

	//save hog txt and set hog
	genHogArg();
	cout << "ģ���ѱ��浽test2.xml" << endl;
	hog.setSVMDetector(myDetector);
	cout << "hog��ʹ��re-train ��SVM����\n ���Կ�ʼ�����..." << endl;

}

void Detector::testOnTestPosDataSet()
{
	int rightNum = 0;

	for (int i = 0; i < data.testPosImgList.size(); i++) {
		cout << "��ʼ���..."<<i << endl;
		Mat src = imread(data.testPosImgList[i]);
		src = src(Rect(3, 3, 64, 128));
		vector<float> descriptor;
		hog.compute(src, descriptor, Size(8, 8));
		Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//����������������������
		for (int j = 0; j < descriptor.size(); j++)
			testFeatureMat.at<float>(0, j) = descriptor[j];
		double result = svm->predict(testFeatureMat);//�������
		if (result >= 1.0)
			rightNum++;
		cout << "�����" << result << endl;
	}
	cout << "������ϣ�" << endl;
	float precision = float(rightNum) / data.testPosImgList.size();
	cout << "TP=" << rightNum << endl;
	cout << "���������Լ��ϵ�׼ȷ�ʣ�" << precision << endl;
}

void Detector::testOnTestNegDataSet()
{
	int rightNum = 0;

	for (int i = 0; i < data.testNegImgList.size(); i++) {
		cout << "��ʼ���..." << i << endl;
		Mat src = imread(data.testNegImgList[i]);
		vector<float> descriptor;
		hog.compute(src, descriptor, Size(8, 8));
		Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//����������������������
		for (int j = 0; j < descriptor.size(); j++)
			testFeatureMat.at<float>(0, j) = descriptor[j];
		double result = svm->predict(testFeatureMat);//�������
		if (result <= -1.0)
			rightNum++;
		cout << "�����" << result << endl;
	}
	cout << "������ϣ�" << endl;
	float precision = float(rightNum) / data.testNegImgList.size();
	cout << "TN=" << rightNum << endl;
	cout << "���������Լ��ϵ�׼ȷ�ʣ�" << precision << endl;
}

void Detector::testPic() {
	Mat srcc = imread("timgRD981YXB.jpg");
	Mat src = srcc.clone();
	vector<Rect> found, found_filtered;//���ο�����
	cout << "���ж�߶�HOG������" << endl;
	hog.detectMultiScale(src, found, 0, Size(8, 8), Size(16, 16), 1.05, 2);//��ͼƬ���ж�߶����˼��
	cout << "�ҵ��ľ��ο������" << found.size() << endl;

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
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

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
		cout << "���ļ�ʧ��\n";
		return;
	}
	Mat frame;

	for (;; )
	{
		cap >> frame;
		if (frame.empty())
			break;
		vector<Rect> found, found_filtered;//���ο�����
										   //cout << "���ж�߶�HOG������" << endl;
		hog.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//��ͼƬ���ж�߶����˼��
																				   //cout << "�ҵ��ľ��ο������" << found.size() << endl;

																				   //�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
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

		//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
		cout << "����֡" << endl;
	}
	cap.release();
}

void Detector::genHogArg()
{
	//get support vectors
	int DescriptorDim = svm->getVarCount();//����������ά������HOG�����ӵ�ά��
	Mat supportVector = svm->getSupportVectors();
	int supportVectorNum = supportVector.rows;//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	//get rho
	vector<float> svm_alpha;
	vector<float> svm_svidx;
	float svm_rho;
	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);

	//get resultMat
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha�����������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������ÿ��һ��֧������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ�� ������

	supportVectorMat = supportVector;

	//--��alpha���������ݸ��Ƶ�alphaMat��
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = svm_alpha[i];
	}

	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm_rho);
	cout << "�����ά����" << myDetector.size() << endl;

	//�������Ӳ������ļ�
	ofstream fout("test.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}
	cout << "���������txt�ɹ���" << endl;
}

vector<float> Detector::loadHogTxt(string hogTxt)
{
	ifstream fin_detector(hogTxt);//���Լ�ѵ����SVM������ļ�
	vector<float>myDetector;

	while (!fin_detector.eof())
	{
		float temp;
		fin_detector >> temp;
		myDetector.push_back(temp);//������������
	}
	cout << "�����ά����" << myDetector.size() << endl;

	return myDetector;
}
