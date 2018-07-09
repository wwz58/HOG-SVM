#ifndef BODYDETECTOR_H
#define BODYDETECTOR_H

#include <QWidget>
#include <QTimer>
#include <QMessageBox>
#include <QFileDialog>
#include "opencv2/ml/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>

using namespace cv;
using namespace std;

namespace Ui {
class BodyDetector;
}

class BodyDetector : public QWidget
{
    Q_OBJECT

public:
    explicit BodyDetector(QWidget *parent = 0);
    ~BodyDetector();

private slots:
    void on_openFile_clicked();

    void on_process_clicked();

    void processNextFrame();

    void nextFrame();

    void detect(Mat& img,vector<Rect>& found_filtered);

private:
    Ui::BodyDetector *ui;
    QImage  Mat2QImage(Mat cvImg);

    QString qtFilename;
    string filename;

    Mat img;
    QImage qtImg;

    VideoCapture capture;
    double rate;
    QTime *timer;

    HOGDescriptor myHOG;

};

#endif // BODYDETECTOR_H
