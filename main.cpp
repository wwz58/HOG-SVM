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

#define PosSamNO 2416    //正样本个数
//#define PosMITNo 572   //扩充的正样本+NICTA7892 1000



#define NegSamNO 12180   //负样本个数12180

#define TRAIN false   //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体

#define TEST false
//HardExampleNo：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值
#define HardExampleNO 4410//8116   //4524 5866 6215 7801  5491
#define HARD_EXAMPLE false     //是否生成hardExample，true时从负样本原图中生成hardExample
#define TestNO 2//288    //测试集样本数
#define PICTURE true   //处理的为图片集还是视频 true为图片

/*
//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public ml::SVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double get_svm_rho()
	{
		return this->getDecisionFunction(0, svm_alpha, svm_svidx);
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	vector<float> svm_alpha;
	vector<float> svm_svidx;
	float svm_rho;

};
*/


int main()
{
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	Ptr<ml::SVM>svm = ml::SVM::create();//SVM分类器

										//若TRAIN为true，重新训练分类器
	if (TRAIN)
	{
		cout << "开始处理训练图片" << endl;
		string ImgName;//图片名(绝对路径)
		ifstream finPos("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\pos\\trainPos.txt");//正样本图片的文件名列表
																		//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\croppedNeg\\croppedImgList.txt");//负样本图片的文件名列表
		//ifstream finMITPos("E:\\target detection documents\\MITPos.txt");//扩充MIT图片的文件名列表



		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num<PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			//ImgName = "D:\\DataSet\\PersonFromVOC2012\\" + ImgName;//加上正样本的路径名
			//ImgName = "E:\\target detection documents\\data\\INRIADATA\\normalized_images\\train\\pos\\" + ImgName;//加上正样本的路径名
			ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\pos\\" + ImgName;
			Mat src = imread(ImgName);//读取图片
			if (CENTRAL_CROP)
				src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			    //src = src(Rect(0, 0, 64, 64));		   						 //resize(src,src,Size(64,128));把人像裁去下半部分

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;

													  //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
				cout<<"描述子维数："<< DescriptorDim <<endl;
												   //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//sampleFeatureMat = Mat::zeros(PosSamNO + PosMITNo + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
				//sampleLabelMat = Mat::zeros(PosSamNO + PosMITNo + NegSamNO + HardExampleNO, 1, CV_32SC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
		}
		/*
		//依次读取补充的MIT正样本图片，生成HOG描述子
		for (int num = 0; num<PosMITNo && getline(finMITPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "E:\\target detection documents\\data upper\\c\\" + ImgName;//加上补充样本的路径名
			Mat src = imread(ImgName);//读取图片
									  //resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
													  //cout<<"描述子维数："<<descriptors.size()<<endl;

													  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = 1;//补充正样本类别为1，有人
		}
		*/


		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num<NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\croppedNeg\\" + ImgName;
			//ImgName = "E:\\target detection documents\\data\\neg\\" + ImgName;//加上负样本的路径名
			Mat src = imread(ImgName);//读取图片
									  //resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
													  //cout<<"描述子维数："<<descriptors.size()<<endl;

													  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				//sampleFeatureMat.at<float>(num + PosSamNO + PosMITNo, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人
			//sampleLabelMat.at<float>(num + PosSamNO + PosMITNo, 0) = -1;//负样本类别为-1，无人
		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			//ifstream finHardExample("E:\\target detection documents\\data upper\\hardExample.txt");//HardExample负样本的文件名列表
			String hardPattern("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\hard\\*jpg");
			vector<String> hardImgs;
			glob(hardPattern, hardImgs, false);
			//依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num<HardExampleNO; num++)
			{
				cout << "处理：hard" << endl;
				//ImgName = "E:\\target detection documents\\data\\INRIADATA\\normalized_images\\train\\hardExample\\" + ImgName;
				//ImgName = "E:\\target detection documents\\data upper\\childhard\\" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(hardImgs[num]);//读取图片
										  //resize(src,img,Size(64,128));

				vector<float> descriptors;//HOG描述子向量
				hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
														  //cout<<"描述子维数："<<descriptors.size()<<endl;

														  //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
					//sampleFeatureMat.at<float>(num + PosSamNO + PosMITNo + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
				//sampleLabelMat.at<float>(num + PosSamNO + PosMITNo + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}

		////输出样本的HOG特征向量矩阵到文件
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//	fout<<i<<endl;
		//	for(int j=0; j<DescriptorDim; j++)
		//		fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//	fout<<endl;
		//}

		//训练SVM分类器

		svm->setType(ml::SVM::C_SVC);
		svm->setKernel(ml::SVM::LINEAR);
		svm->setGamma(1);
		svm->setC(0.01);
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代 
		//TermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON));
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		//CvSVMParams param(cv::ml::SVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

		cout << "开始训练SVM分类器" << endl;
		//svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
		svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);
		cout << "训练完成" << endl;
		svm->save("svmLinearAllHard.xml");//将训练好的SVM模型保存为xml文件

	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		//svm=ml::SVM::load("SVM_HOG_2400PosINRIA_12000Neg_HardExample(误报少了漏检多了).xml");//从XML文件读取训练好的SVM模型
		svm = ml::SVM::load("svmLinearAllHard.xml");
	}


	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数
	Mat supportVector = svm->getSupportVectors();
	int supportVectorNum = supportVector.rows;//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;
	//-------------------------------------------------------------
	vector<float> svm_alpha;
	vector<float> svm_svidx;
	float svm_rho;
	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);
	//----------------------------------------------------------------
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha行向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵，每行一个支持向量
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果 行向量

	//将支持向量的数据复制到supportVectorMat矩阵中
	supportVectorMat = supportVector;
	/*for (int i = 0; i<supportVectorNum; i++)
	{
	const float * pSVData = svm->getSupportVectors(i);//返回第i个支持向量的数据指针
	for (int j = 0; j<DescriptorDim; j++)
	{
	//cout<<pData[j]<<" ";
	supportVectorMat.at<float>(i, j) = pSVData[j];
	}
	}*/

	//将alpha向量的数据复制到alphaMat中
	//double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = svm_alpha[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm_rho);
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//保存检测子参数到文件
	//ofstream fout("hogDetectorLinearNoHard.txt");
	ofstream fout("hogDetectorAllHard.txt");
	for (int i = 0; i<myDetector.size(); i++)
	{
		fout << myDetector[i] << endl;
	}
	cout << "保存参数成功！" << endl;
	/************************* hardExample的生成***********************************************/
#if (0)
	{
		int hardExampleCount = 0;// 7058; //  7058;4524;
		Mat src;
		char saveName[256];//剪裁出来的hard example图片的文件名
		string ImgName;
	
		ifstream fin_imgList("E:\\target detection documents\\origNegu.txt");//打开原始负样本图片文件列表
		while (getline(fin_imgList, ImgName))
		{
			cout << "处理：" << ImgName << endl;
			string fullName = "E:\\target detection documents\\data\\INRIADATA\\original_images\\train\\neg\\" + ImgName;//加上路径名
			src = imread(fullName);//读取图片
			Mat img = src.clone();//复制原图

			vector<Rect> found;//矩形框数组
							   //对负样本原图进行多尺度检测，检测出的都是误报
			myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
			//遍历从图像中检测出来的矩形框，得到hard example
			for (int i = 0; i < found.size(); i++)
			{
				//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部
				Rect r = found[i];
				if (r.x < 0)
					r.x = 0;
				if (r.y < 0)
					r.y = 0;
				if (r.x + r.width > src.cols)
					r.width = src.cols - r.x;
				if (r.y + r.height > src.rows)
					r.height = src.rows - r.y;

				//将矩形框保存为图片，就是Hard Example
				Mat hardExampleImg = src(r);//从原图上截取矩形框大小的图片
				resize(hardExampleImg, hardExampleImg, Size(64, 64));//将剪裁出来的图片缩放为64*128大小
				sprintf(saveName, "hard%09d.jpg", hardExampleCount++);//生成hard example图片的文件名
				imwrite(saveName, hardExampleImg);//保存文件
												  //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
												  //r.x += cvRound(r.width*0.1);
												  //r.width = cvRound(r.width*0.8);
												  //r.y += cvRound(r.height*0.07);
												  //r.height = cvRound(r.height*0.8);
				rectangle(img, r.tl(), r.br(), Scalar(0, 255, 0), 3);

			}																			  //ifstream fin_imgList("subset.txt");
		}
	}
#endif

	/**************读入图片进行HOG行人检测******************/
	//Mat src = imread("timgRD981YXB.jpg");
	//Mat src = imread("2007_000423.jpg");
	if (PICTURE)
	{
		/*
		string ImgName;
		ifstream finTest("E:\\target detection documents\\testPos.txt");//HardExample负样本的文件名列表
		//ifstream finTest("E:\\target detection documents\\data upper\\test.txt");
		for (int num = 0; num < TestNO && getline(finTest, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			ImgName = "E:\\target detection documents\\data\\INRIADATA\\original_images\\test\\pos\\" + ImgName;//加上测试样本的路径名
			//ImgName = "E:\\target detection documents\\data upper\\test\\" + ImgName;
			Mat src = imread(ImgName);//读取图片
			*/
		Mat srcc = imread("timgRD981YXB.jpg");
		Mat src = srcc.clone();
			vector<Rect> found, found_filtered;//矩形框数组
			cout << "进行多尺度HOG人体检测" << endl;
			myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(16, 16), 1.05, 2);//对图片进行多尺度行人检测
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
			/*
		}
		*/
	}    //注意：imshow之后必须加waitKey，否则无法显示图像

		 //PICTURE为false则对视频进行处理

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
			vector<Rect> found, found_filtered;//矩形框数组
											   //cout << "进行多尺度HOG人体检测" << endl;
			myHOG.detectMultiScale(frame, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);//对图片进行多尺度行人检测
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

	if (TEST) {
		/******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
		//读取测试图片(64*128大小)，并计算其HOG描述子
		//Mat testImg = imread("person014142.jpg");
		//Mat testImg = imread("noperson000026.jpg");
		//String pattern("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\train\\croppedNeg\\*.jpg");
		String pattern("C:\\Users\\20203\\Downloads\\INRIADATA\\normalized_images\\test\\pos\\*.png");
		vector<String> imgs;
		glob(pattern, imgs, false);
		int rightNum = 0;
		int dataNo = 20;
		for (int i = 0; i < imgs.size(); i++) {
			cout << "处理" << static_cast<std::string>(imgs[i]) << endl;
			vector<float> descriptor;
			Mat testImg = imread(imgs[i]);
			testImg = testImg(Rect(3, 3, 64, 128));
			hog.compute(testImg, descriptor, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
			Mat testFeatureMat = Mat::zeros(1, descriptor.size(), CV_32FC1);//测试样本的特征向量矩阵
															   //将计算好的HOG描述子复制到testFeatureMat矩阵中
			for (int j = 0; j < descriptor.size(); j++)
				testFeatureMat.at<float>(0, j) = descriptor[j];

			//用训练好的SVM分类器对测试图片的特征向量进行分类
			double result = svm->predict(testFeatureMat);//返回类标
			if (result >= 1.0)
				rightNum++;
			cout << "分类结果：" << result << endl;
		}
		cout << "处理完毕！" << endl;
		float precision = float(rightNum) / imgs.size();
		cout << "TP=" << rightNum << endl;
		cout << "正样本测试集上的准确率：" << precision << endl;
		/*
		vector<float> descriptor;
		hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
		Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
		//将计算好的HOG描述子复制到testFeatureMat矩阵中
		for(int i=0; i<descriptor.size(); i++)
			testFeatureMat.at<float>(0,i) = descriptor[i];

		//用训练好的SVM分类器对测试图片的特征向量进行分类
		int result = svm->predict(testFeatureMat);//返回类标
		cout<<"分类结果："<<result<<endl;
		*/
	}
	system("pause");
}