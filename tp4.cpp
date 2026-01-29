#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;


Mat transpose(Mat image)
{
    Mat res = Mat::zeros(image.cols,image.rows,CV_32FC1);
    image.convertTo(image,CV_32FC1);
    for(int x=0;x<image.rows;x++){
        for(int y=0;y<image.cols;y++){
            res.at<float>(y,x)= image.at<float>(x,y);
        }
    }

    return res;
}


float interpolate_nearest(Mat image, float x, float y)
{
    float v=0;
    int nv_x,nv_y;
   
    if (int(x+0.5f) >= image.rows){nv_x=image.rows-1;}
    else if (int(x+0.5f) < 0 ){nv_x=0;}
    else{nv_x=int(x+0.5f);}
    if (int(y+0.5f) >= image.cols){nv_y=image.cols-1;}
    else if (int(y+0.5f) < 0 ){nv_y=0;}
    else{nv_y=int(y+0.5f);}

    v = image.at<float>(nv_x,nv_y);
    return v;

}


/**
    Compute the value of a bilinear interpolation in image Mat at position (x,y)
*/
float interpolate_bilinear(Mat image, float y, float x)
{
    float v=0;
    /********************************************
                YOUR CODE HERE
    *********************************************/
    
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return v;
}


Mat expand(Mat image, int factor, float(* interpolationFunction)(cv::Mat img, float x, float y))
{
    assert(factor>0);
    Mat res = Mat::zeros((image.rows-1)*factor,(image.cols-1)*factor,CV_32FC1);
    Mat tmp=image.clone();
    image.convertTo(tmp,CV_32FC1);

    for(int x=0;x<res.rows;x++){
        for(int y=0;y<res.cols;y++){
            res.at<float>(x,y) = interpolationFunction(tmp,x/factor,y/factor);
        }
    }

    res.convertTo(res,CV_8UC1); 
    return res;
}

/**
    Performs a rotation of the input image with the given angle (clockwise) and the given interpolation method.
    The center of rotation is the center of the image.

    Ouput size depends of the input image size and the rotation angle.

    Output pixels that map outside the input image are set to 0.
*/
Mat rotate(Mat image, float angle, float(* interpolationFunction)(cv::Mat image, float y, float x))
{
    Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
    hint: to determine the size of the output, take
    the bounding box of the rotated corners of the 
    input image.
    *********************************************/
    
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;

}


int main(){    

    string path = "C:\\Users\\user\\OneDrive\\Bureau\\AnUniv\\m1-p\\image\\ImageProcessingLab-main\\pic\\cat.jpg";
    Mat img = imread(path,IMREAD_GRAYSCALE);
    Mat res;
    imshow("image",img);
    waitKey(0);

    ////QST  1
    // res = transpose(img);
    // res.convertTo(res,CV_8UC1);

    ////QST  2
    res = expand(img,2,interpolate_nearest);

    ////QST  3
    
    ////QST  5
    

    imshow("image22",res);
    waitKey(0);
    return 0;
}