#include "bodydetector.h"
#include "ui_bodydetector.h"



BodyDetector::BodyDetector(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::BodyDetector)
{
    ui->setupUi(this);

    ui->previewBox->setScaledContents(true);
    ui->openFile->setEnabled(true);
    ui->process->setEnabled(false);

    /*static vector <float> x = HOGDescriptor::getDefaultPeopleDetector();
    myHOG.setSVMDetector(x);//*/
    ifstream fin_detector("C:\\Users\\20203\\Documents\\BodyDetector\\ourDetector.txt");//打开自己训练的SVM检测器文件
    vector<float> myDetector;//3781维的检测器参数
    float temp;
    while (!fin_detector.eof())
    {
        fin_detector >> temp;
        myDetector.push_back(temp);//放入检测器数组
    }
    //ui->label->setText("加载检测子其维数"+QString::number(myDetector.size())+"维\n");
    myHOG.setSVMDetector(myDetector);//*/
}

BodyDetector::~BodyDetector()
{
    delete ui;
}

QImage BodyDetector::Mat2QImage(cv::Mat cvImg)
{
    QImage qImg;
    if(cvImg.channels()==3)                             //3 channels color image
    {
        cv::cvtColor(cvImg,cvImg,CV_BGR2RGB);
        qImg =QImage((const unsigned char*)(cvImg.data),
                    cvImg.cols, cvImg.rows,
                    cvImg.cols*cvImg.channels(),
                    QImage::Format_RGB888);
    }
    else if(cvImg.channels()==1)                    //grayscale image
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                    cvImg.cols,cvImg.rows,
                    cvImg.cols*cvImg.channels(),
                    QImage::Format_Indexed8);
    }
    else
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                    cvImg.cols,cvImg.rows,
                    cvImg.cols*cvImg.channels(),
                    QImage::Format_RGB888);
    }

    return qImg;

}

void BodyDetector::on_process_clicked()
{
    ui->openFile->setEnabled(false);
    ui->process->setEnabled(true);
    QFileInfo qtFileinfo(qtFilename);
    //处理视频
    if ((qtFileinfo.suffix()=="mp4")||(qtFileinfo.suffix()=="avi")){
        if (capture.isOpened())
            capture.release();
        capture.open(filename);
        if (capture.isOpened())
        {
            capture >> img;
            if (!img.empty())
            {
                //cvtColor( img, img, COLOR_BGR2GRAY );
                //equalizeHist(img,img);
                vector<Rect> found_filtered;
                detect(img,found_filtered);

                //Opencv显示
                namedWindow("src", 0);
                imshow("src", img);

                //Qt显示
                rate= capture.get(CV_CAP_PROP_FPS);
                qtImg = Mat2QImage(img);
                ui->previewBox->setPixmap(QPixmap::fromImage(qtImg));
                QTimer *timer = new QTimer(this);
                timer->setInterval(1000/(rate*1.5));   //set timer match with FPS
                connect(timer, SIGNAL(timeout()), this, SLOT(processNextFrame()));
                timer->start();
            }
        }
        else{
            return;
        }
    }

    //处理图片
    else if((qtFileinfo.suffix()=="jpg")||(qtFileinfo.suffix()=="png")){
        img = imread(filename);
        //检测
        vector<Rect> found_filtered;
        detect(img,found_filtered);

        //Opencv显示
        namedWindow("src", 0);
        imshow("src", img);

        //Qt显示
        qtImg = Mat2QImage(img);
        ui->previewBox->setPixmap(QPixmap::fromImage(qtImg));
        ui->openFile->setEnabled(true);
        ui->process->setEnabled(false);
    }
}

void BodyDetector::on_openFile_clicked()
{
    ui->openFile->setEnabled(false);
    ui->process->setEnabled(true);
    //获取文件
    qtFilename = QFileDialog::getOpenFileName(
                this,
                tr("Open File"),
                "",
                tr("Image Files(*.jpg *.png);;Video Files(*.mp4 *.avi)")
                );

    filename = qtFilename.toStdString();
    QFileInfo qtFileinfo(qtFilename);

    //源视频文件，展示
    if ((qtFileinfo.suffix()=="mp4")||(qtFileinfo.suffix()=="avi")){
        if(capture.isOpened())
            capture.release();
        capture.open(filename);
        if (capture.isOpened())
        {
            rate= capture.get(CV_CAP_PROP_FPS);         
            capture >> img;
            if (!img.empty())
            {
                qtImg = Mat2QImage(img);
                ui->previewBox->setPixmap(QPixmap::fromImage(qtImg));
                QTimer *timer = new QTimer(this);
                timer->setInterval(1000/(rate*1.5));   //set timer match with FPS
                connect(timer, SIGNAL(timeout()), this, SLOT(nextFrame()));
                timer->start();
            }
        }
    }
    //源图像文件，展示
    else if((qtFileinfo.suffix()=="jpg")||(qtFileinfo.suffix()=="png")){
        img = imread(filename);
        imshow("cvImg",img);
        qtImg = Mat2QImage(img);

        ui->previewBox->setPixmap(QPixmap::fromImage(qtImg));
        ui->process->setEnabled(true);
    }


}

void BodyDetector::processNextFrame(){
    capture >> img;
    if (!img.empty()){
        //cvtColor( img, img, COLOR_BGR2GRAY );
        //equalizeHist(img,img);
        //检测
        vector<Rect> found_filtered;
        detect(img,found_filtered);


        //Opencv显示
        namedWindow("src", 0);
        imshow("src", img);

        //Qt显示
        qtImg=Mat2QImage(img);
        ui->previewBox->setPixmap(QPixmap::fromImage(qtImg));
    }
    else{
        ui->process->setEnabled(false);
        ui->openFile->setEnabled(true);
    }
}

void BodyDetector::nextFrame(){

    capture>>img;
    if(!img.empty())
    {
        qtImg=Mat2QImage(img);
        ui->previewBox->setPixmap(QPixmap::fromImage(qtImg));       
    }
}

void BodyDetector::detect(Mat& img,vector<Rect>& found_filtered){
    //检测
    vector<Rect> found;//矩形框数组
    myHOG.detectMultiScale(img, found, 0, Size(8, 8), Size(16, 16), 1.05, 2);//对图片进行多尺度行人检测
    ui->label->setText("检测到"+QString::number(found.size())+"个目标框\n");

    //处理嵌套框
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
    ui->label->setText("处理后："+QString::number(found.size())+"个目标框\n");

    //缩小，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
    for (int i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.15);
        r.width = cvRound(r.width*0.7);
        r.y += cvRound(r.height*0.1);
        r.height = cvRound(r.height*0.8);
        rectangle(img, r.tl(), r.br(), Scalar(255, 0, 0), 3);
    }
}
