#include <iostream>
#include <fstream>
#include <stdlib.h> //srand()��rand()����
#include <time.h> //time()����
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;



int main()
{
	int CropImageCount = 0; //�ü������ĸ�����ͼƬ����
	Mat src;
	string ImgName;
	char saveName[256];//�ü������ĸ�����ͼƬ�ļ���
	//ifstream fin("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\neg\\INRIANegativeImageList.txt");//��ԭʼ������ͼƬ�ļ��б�
	//ifstream fin("subset.txt");
	ifstream fin("C:\\Users\\20203\\Downloads\\INRIADATA\\original_images\\test\\neg\\testNegList.txt");

	//һ��һ�ж�ȡ�ļ��б�
	while (getline(fin, ImgName))
	{
		cout << "����" << ImgName << endl;
		ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\original_images\\test\\neg\\" + ImgName;
		//ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;
		src = imread(ImgName);
		cout<<"��"<<src.cols<<"���ߣ�"<<src.rows<<endl;

		//ͼƬ��СӦ���������ٰ���һ��64*128�Ĵ���
		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//�������������

			//��ÿ��ͼƬ������ü�10��64*128��С�Ĳ������˵ĸ�����
			for (int i = 0; i<10; i++)
			{
				int x = (rand() % (src.cols - 64)); //���Ͻ�x����
				int y = (rand() % (src.rows - 128)); //���Ͻ�y����
													 //cout<<x<<","<<y<<endl;
				Mat imgROI = src(Rect(x, y, 64, 128));
				sprintf_s(saveName, "noperson%06d.jpg", ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���
				imwrite("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\test\\neg\\"+String(saveName), imgROI);//�����ļ�

				//imwrite(saveName, imgROI);//�����ļ�
			}
		}
	}

	system("pause");
}
