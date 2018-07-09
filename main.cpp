#include "bodydetector.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    BodyDetector w;
    w.show();

    return a.exec();
}
