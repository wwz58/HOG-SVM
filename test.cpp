#include "Detector.h"
using namespace cv;
using namespace std;

int main(int argc,char* argv) {
	//DataSet data= DataSet();
	//data.showInfo();

	//1.train and test
		Detector obj("train");

		//obj.firstTrain();
		//obj.testOnTestPosDataSet();
		
		//obj.reTrain();
		//obj.testOnTestPosDataSet();
		//obj.testOnTestNegDataSet();

	//2.Detector("test") load a trained model
		//Detector obj("test");

		//obj.testOnTestPosDataSet();
		//obj.testOnTestNegDataSet();
		//obj.testPic();
		//obj.testVideo();
		

	/*VideoCapture cap("C:\\Users\\20203\\Desktop\\video.mp4");
	Mat frame;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			cout << "process complete" << endl;
			break;
		}
		else {
			vector<Rect> found, found_filtered;

			hog.detectMultiScale(frame,found,0,Size(8,8),Size(16,16));
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
				rectangle(frame, r.tl(), r.br(), Scalar(255, 0, 0), 3);
			}
			imshow("src", frame);
			waitKey(100);
		}
	}*/
		getchar();
}