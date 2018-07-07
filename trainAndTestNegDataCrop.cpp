#include <iostream>
#include <fstream>
#include <stdlib.h> //srand()和rand()函数
#include <time.h> //time()函数
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;



int main()
{
	int CropImageCount = 0; //裁剪出来的负样本图片个数
	Mat src;
	string ImgName;
	char saveName[256];//裁剪出来的负样本图片文件名
	//ifstream fin("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\neg\\INRIANegativeImageList.txt");//打开原始负样本图片文件列表
	//ifstream fin("subset.txt");
	ifstream fin("C:\\Users\\20203\\Downloads\\INRIADATA\\original_images\\test\\neg\\testNegList.txt");

	//一行一行读取文件列表
	while (getline(fin, ImgName))
	{
		cout << "处理：" << ImgName << endl;
		ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\original_images\\test\\neg\\" + ImgName;
		//ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\neg\\" + ImgName;
		src = imread(ImgName);
		cout<<"宽："<<src.cols<<"，高："<<src.rows<<endl;

		//图片大小应该能能至少包含一个64*128的窗口
		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//设置随机数种子

			//从每张图片中随机裁剪10个64*128大小的不包含人的负样本
			for (int i = 0; i<10; i++)
			{
				int x = (rand() % (src.cols - 64)); //左上角x坐标
				int y = (rand() % (src.rows - 128)); //左上角y坐标
													 //cout<<x<<","<<y<<endl;
				Mat imgROI = src(Rect(x, y, 64, 128));
				sprintf_s(saveName, "noperson%06d.jpg", ++CropImageCount);//生成裁剪出的负样本图片的文件名
				imwrite("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\test\\neg\\"+String(saveName), imgROI);//保存文件

				//imwrite(saveName, imgROI);//保存文件
			}
		}
	}

	system("pause");
}
