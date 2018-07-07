#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

int hardExampleCount = 0; //hard example����

int main()
{
	Mat src;
	char saveName[256];//���ó�����hard exampleͼƬ���ļ���
	string ImgName;
	ifstream fin_detector("hogDetectorLinearNoHard.txt");//���Լ�ѵ����SVM������ļ�
	ifstream fin_imgList("C:\\Users\\20203\\Downloads\\INRIADATA\\original_images\\train\\neg\\trainOriginNegList.txt");//��ԭʼ������ͼƬ�ļ��б�
													   //ifstream fin_imgList("subset.txt");

													   //���ļ��ж����Լ�ѵ����SVM����
	float temp;
	vector<float> myDetector;//3781ά�ļ��������
	while (!fin_detector.eof())
	{
		fin_detector >> temp;
		myDetector.push_back(temp);//������������
	}
	cout << "�����ά����" << myDetector.size() << endl;

	//namedWindow("src",0);
	HOGDescriptor hog;//HOG���������
	hog.setSVMDetector(myDetector);//���ü��������Ϊ�Լ�ѵ����SVM����

								   //һ��һ�ж�ȡ�ļ��б�
	while (getline(fin_imgList, ImgName))
	{
		cout << "����" << ImgName << endl;
		string fullName = "C:\\Users\\20203\\Downloads\\INRIADATA\\original_images\\train\\neg\\" + ImgName;//����·����
		src = imread(fullName);//��ȡͼƬ
		Mat img = src.clone();//����ԭͼ

		vector<Rect> found;//���ο�����
		//cout << found.size() << endl;

						   //�Ը�����ԭͼ���ж�߶ȼ�⣬�����Ķ�����
		hog.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		cout <<"��������"<< found.size() << endl;

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
				resize(hardExampleImg, hardExampleImg, Size(64, 128), INTER_CUBIC);//�����ó�����ͼƬ����Ϊ64*128��С
				sprintf_s(saveName, "hardexample%09d.jpg", hardExampleCount++);//����hard exampleͼƬ���ļ���
				imwrite("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\hard\\" + String(saveName), hardExampleImg);//�����ļ�


												  //�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
												  //r.x += cvRound(r.width*0.1);
												  //r.width = cvRound(r.width*0.8);
												  //r.y += cvRound(r.height*0.07);
												  //r.height = cvRound(r.height*0.8);
				rectangle(img, r.tl(), r.br(), Scalar(0, 255, 0), 3);

			}
		//imwrite(ImgName,img);
		imshow("img", img);
		waitKey(10);//ע�⣺imshow֮��һ��Ҫ��waitKey�������޷���ʾͼ��
	}

	system("pause");
}
