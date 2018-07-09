#-------------------------------------------------
#
# Project created by QtCreator 2018-07-04T13:07:44
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = BodyDetector
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        bodydetector.cpp

HEADERS += \
        bodydetector.h

FORMS += \
        bodydetector.ui

INCLUDEPATH += D:\opencv320x64contrib\include

LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_core320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_highgui320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_imgcodecs320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_imgproc320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_line_descriptor320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_ml320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_objdetect320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_video320.lib
LIBS += D:\opencv320x64contrib\x64\vc14\lib\opencv_videoio320.lib

RESOURCES += \
    mytrainedhogdescriptor.qrc
