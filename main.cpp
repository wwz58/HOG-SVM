#pragma warning(disable:4996)
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>

using namespace std;
using namespace cv;

#define PosSamNO 2416    //����������
//#define PosMITNo 572   //�����������+NICTA7892 1000



#define NegSamNO 12180   //����������12180

#define TRAIN false   //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define CENTRAL_CROP true   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����

#define TEST false
//HardExampleNo�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ
#define HardExampleNO 4410//8116   //4524 5866 6215 7801  5491
#define HARD_EXAMPLE false     //�Ƿ�����hardExample��trueʱ�Ӹ�����ԭͼ������hardExample
#define TestNO 2//288    //���Լ�������
#define PICTURE true   //�����ΪͼƬ��������Ƶ trueΪͼƬ

/*
//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������
class MySVM : public ml::SVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double get_svm_rho()
	{
		return this->getDecisionFunction(0, svm_alpha, svm_svidx);
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	vector<float> svm_alpha;
	vector<float> svm_svidx;
	float svm_rho;

};
*/


int main()
{
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	Ptr<ml::SVM>svm = ml::SVM::create();//SVM������

										//��TRAINΪtrue������ѵ��������
	if (TRAIN)
	{
		cout << "��ʼ����ѵ��ͼƬ" << endl;
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\pos\\trainPos.txt");//������ͼƬ���ļ����б�
																		//ifstream finPos("PersonFromVOC2012List.txt");//������ͼƬ���ļ����б�
		ifstream finNeg("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\croppedNeg\\croppedImgList.txt");//������ͼƬ���ļ����б�
		//ifstream finMITPos("E:\\target detection documents\\MITPos.txt");//����MITͼƬ���ļ����б�



		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����


		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//������������·����
			//ImgName = "E:\\target detection documents\\data\\INRIADATA\\normalized_images\\train\\pos\\" + ImgName;//������������·����
			ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\pos\\" + ImgName;
			Mat src = imread(ImgName);//��ȡͼƬ
			if (CENTRAL_CROP)
				src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
			    //src = src(Rect(0, 0, 64, 64));		   						 //resize(src,src,Size(64,128));�������ȥ�°벿��

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			//cout<<"������ά����"<<descriptors.size()<<endl;

													  //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
				cout<<"������ά����"<< DescriptorDim <<endl;
												   //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//sampleFeatureMat = Mat::zeros(PosSamNO + PosMITNo + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
				//sampleLabelMat = Mat::zeros(PosSamNO + PosMITNo + NegSamNO + HardExampleNO, 1, CV_32SC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
		}
		/*
		//���ζ�ȡ�����MIT������ͼƬ������HOG������
		for (int num = 0; num<PosMITNo && getline(finMITPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "E:\\target detection documents\\data upper\\c\\" + ImgName;//���ϲ���������·����
			Mat src = imread(ImgName);//��ȡͼƬ
									  //resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
													  //cout<<"������ά����"<<descriptors.size()<<endl;

													  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO, 0) = 1;//�������������Ϊ1������
		}
		*/


		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\croppedNeg\\" + ImgName;
			//ImgName = "E:\\target detection documents\\data\\neg\\" + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
									  //resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
													  //cout<<"������ά����"<<descriptors.size()<<endl;

													  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				//sampleFeatureMat.at<float>(num + PosSamNO + PosMITNo, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//���������Ϊ-1������
			//sampleLabelMat.at<float>(num + PosSamNO + PosMITNo, 0) = -1;//���������Ϊ-1������
		}

		//����HardExample������
		if (HardExampleNO > 0)
		{
			//ifstream finHardExample("E:\\target detection documents\\data upper\\hardExample.txt");//HardExample���������ļ����б�
			String hardPattern("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\hard\\*jpg");
			vector<String> hardImgs;
			glob(hardPattern, hardImgs, false);
			//���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num<HardExampleNO; num++)
			{
				cout << "����hard" << endl;
				//ImgName = "E:\\target detection documents\\data\\INRIADATA\\normalized_images\\train\\hardExample\\" + ImgName;
				//ImgName = "E:\\target detection documents\\data upper\\childhard\\" + ImgName;//����HardExample��������·����
				Mat src = imread(hardImgs[num]);//��ȡͼƬ
										  //resize(src,img,Size(64,128));

				vector<float> descriptors;//HOG����������
				hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
														  //cout<<"������ά����"<<descriptors.size()<<endl;

														  //������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
					//sampleFeatureMat.at<float>(num + PosSamNO + PosMITNo + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
				//sampleLabelMat.at<float>(num + PosSamNO + PosMITNo + NegSamNO, 0) = -1;//���������Ϊ-1������
			}
		}

		////���������HOG�������������ļ�
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//	fout<<i<<endl;
		//	for(int j=0; j<DescriptorDim; j++)
		//		fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//	fout<<endl;
		//}

		//ѵ��SVM������

		svm->setType(ml::SVM::C_SVC);
		svm->setKernel(ml::SVM::LINEAR);
		svm->setGamma(1);
		svm->setC(0.01);
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ���� 
		//TermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON));
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		//CvSVMParams param(cv::ml::SVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

		cout << "��ʼѵ��SVM������" << endl;
		//svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
		svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);
		cout << "ѵ�����" << endl;
		svm->save("svmLinearAllHard.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	}
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		//svm=ml::SVM::load("SVM_HOG_2400PosINRIA_12000Neg_HardExample(������©�����).xml");//��XML�ļ���ȡѵ���õ�SVMģ��
		svm = ml::SVM::load("svmLinearAllHard.xml");
	}


	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	DescriptorDim = svm->getVarCount();//����������ά������HOG�����ӵ�ά��
	Mat supportVector = svm->getSupportVectors();
	int supportVectorNum = supportVector.rows;//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;
	//-------------------------------------------------------------
	vector<float> svm_alpha;
	vector<float> svm_svidx;
	float svm_rho;
	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);
	//----------------------------------------------------------------
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha�����������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������ÿ��һ��֧������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ�� ������

	//��֧�����������ݸ��Ƶ�supportVectorMat������
	supportVectorMat = supportVector;
	/*for (int i = 0; i<supportVectorNum; i++)
	{
	const float * pSVData = svm->getSupportVectors(i);//���ص�i��֧������������ָ��
	for (int j = 0; j<DescriptorDim; j++)
	{
	//cout<<pData[j]<<" ";
	supportVectorMat.at<float>(i, j) = pSVData[j];
	}
	}*/

	//��alpha���������ݸ��Ƶ�alphaMat��
	//double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = svm_alpha[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm_rho);
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	//ofstream fout("hogDetectorLinearNoHard.txt");
	ofstream fout("hogDetectorAllHard.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}
	cout << "��������ɹ���" << endl;
	/************************* hardExample������***********************************************/
#if (0)
	{
		int hardExampleCount = 0;// 7058; //  7058;4524;
		Mat src;
		char saveName[256];//���ó�����hard exampleͼƬ���ļ���
		string ImgName;
	
		ifstream fin_imgList("E:\\target detection documents\\origNegu.txt");//��ԭʼ������ͼƬ�ļ��б�
		while (getline(fin_imgList, ImgName))
		{
			cout << "����" << ImgName << endl;
			string fullName = "E:\\target detection documents\\data\\INRIADATA\\original_images\\train\\neg\\" + ImgName;//����·����
			src = imread(fullName);//��ȡͼƬ
			Mat img = src.clone();//����ԭͼ

			vector<Rect> found;//���ο�����
							   //�Ը�����ԭͼ���ж�߶ȼ�⣬�����Ķ�����
			myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
			//������ͼ���м������ľ��ο򣬵õ�hard example
			for (int i = 0; i < found.size(); i++)
			{
				//�������ĺܶ���ο򶼳�����ͼ��߽磬����Щ���ο�ǿ�ƹ淶��ͼ��߽��ڲ�
				Rect r = found[i];
				if (r.x < 0)
					r.x = 0;
				if (r.y < 0)
					r.y = 0;
				if (r.x + r.width > src.cols)
					r.width = src.cols - r.x;
				if (r.y + r.height > src.rows)
					r.height = src.rows - r.y;

				//�����ο򱣴�ΪͼƬ������Hard Example
				Mat hardExampleImg = src(r);//��ԭͼ�Ͻ�ȡ���ο��С��ͼƬ
				resize(hardExampleImg, hardExampleImg, Size(64, 64));//�����ó�����ͼƬ����Ϊ64*128��С
				sprintf(saveName, "hard%09d.jpg", hardExampleCount++);//����hard exampleͼƬ���ļ���
				imwrite(saveName, hardExampleImg);//�����ļ�
												  //�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
												  //r.x += cvRound(r.width*0.1);
												  //r.width = cvRound(r.width*0.8);
												  //r.y += cvRound(r.height*0.07);
												  //r.height = cvRound(r.height*0.8);
				rectangle(img, r.tl(), r.br(), Scalar(0, 255, 0), 3);

			}																			  //ifstream fin_imgList("subset.txt");
		}
	}
#endif

	/**************����ͼƬ����HOG���˼��******************/
	//Mat src = imread("timgRD981YXB.jpg");
	//Mat src = imread("2007_000423.jpg");
	if (PICTURE)
	{
		/*
		string ImgName;
		ifstream finTest("E:\\target detection documents\\testPos.txt");//HardExample���������ļ����б�
		//ifstream finTest("E:\\target detection documents\\data upper\\test.txt");
		for (int num = 0; num < TestNO && getline(finTest, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			ImgName = "E:\\target detection documents\\data\\INRIADATA\\original_images\\test\\pos\\" + ImgName;//���ϲ���������·����
			//ImgName = "E:\\target detection documents\\data upper\\test\\" + ImgName;
			Mat src = imread(ImgName);//��ȡͼƬ
			*/
		Mat srcc = imread("timgRD981YXB.jpg");
		Mat src = srcc.clone();
			vector<Rect> found, found_filtered;//���ο�����
			cout << "���ж�߶�HOG������" << endl;
			myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(16, 16), 1.05, 2);//��ͼƬ���ж�߶����˼��
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
			/*
		}
		*/
	}    //ע�⣺imshow֮������waitKey�������޷���ʾͼ��

		 //PICTUREΪfalse�����Ƶ���д���

	else {
		VideoCapture cap;
		cap.open("C:\\FFOutput\\video.mp4");
		if (!cap.isOpened())
			return -1;
		Mat frame;
		
		for( ;; )
		{
			cap >> frame;
			if (frame.empty())
				break;
			vector<Rect> found, found_filtered;//���ο�����
											   //cout << "���ж�߶�HOG������" << endl;
			myHOG.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//��ͼƬ���ж�߶����˼��
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

	if (TEST) {
		/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
		//��ȡ����ͼƬ(64*128��С)����������HOG������
		//Mat testImg = imread("person014142.jpg");
		//Mat testImg = imread("noperson000026.jpg");
		//String pattern("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\croppedNeg\\*.jpg");
		String pattern("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\test\\pos\\*.png");
		vector<String> imgs;
		glob(pattern, imgs, false);
		int rightNum = 0;
		int dataNo = 20;
		for (int i = 0; i < imgs.size(); i++) {
			cout << "����" << static_cast<std::string>(imgs[i]) << endl;
			vector<float> descriptor;
			Mat testImg = imread(imgs[i]);
			testImg = testImg(Rect(3, 3, 64, 128));
			hog.compute(testImg, descriptor, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//����������������������
															   //������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
			for (int j = 0; j < descriptor.size(); j++)
				testFeatureMat.at<float>(0, j) = descriptor[j];

			//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
			double result = svm->predict(testFeatureMat);//�������
			if (result >= 1.0)
				rightNum++;
			cout << "��������" << result << endl;
		}
		cout << "������ϣ�" << endl;
		float precision = float(rightNum) / imgs.size();
		cout << "TP=" << rightNum << endl;
		cout << "���������Լ��ϵ�׼ȷ�ʣ�" << precision << endl;
		/*
		vector<float> descriptor;
		hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������
		//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
		for(int i=0; i<descriptor.size(); i++)
			testFeatureMat.at<float>(0,i) = descriptor[i];

		//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
		int result = svm->predict(testFeatureMat);//�������
		cout<<"��������"<<result<<endl;
		*/
	}
	system("pause");
}