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



float interpolate_bilinear(Mat image, float x, float y)
{
    float v=0;
    int x1 = int(x),y1 = int(x), x2= int(x)+1, y2 = int(y)+1;
    float alpha,beta;

    alpha = (x - x1)/(x2 - x1);
    beta = (y - y1)/(y2 - y1);

    v = (1-alpha)*(1-beta)*(image.at<float>(x1,y1)) + (alpha)*(1-beta)*(image.at<float>(x2,y1))
     + (1-alpha)*(beta)*(image.at<float>(x1,y2)) + (alpha)*(beta)*(image.at<float>(x2,y2));

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

    return res;
}


Mat rotation(Mat image, float angle, float(* interpolationFunction)(cv::Mat image, float y, float x))
{
    int nv_x,nv_y;
    float rad = (angle * M_PI) / 180;
    Mat tmp;
    image.convertTo(tmp,CV_32FC1);

    int ctr_x =(tmp.rows)/2; 
    int ctr_y = (tmp.cols)/2;
    //bounding box (0,0) , (f,f) , (0,f) , (f,0)
    float max_x = max( {(cos(rad) *(0-ctr_x) - sin(rad) * (0-ctr_y) + ctr_x) , (cos(rad) *((image.rows-1)-ctr_x) - sin(rad) * ((image.cols-1)-ctr_y) + ctr_x)  ,(cos(rad) *(0-ctr_x) - sin(rad) * ((image.cols-1)-ctr_y) + ctr_x) ,(cos(rad) *((image.rows-1)-ctr_x) - sin(rad) * (0-ctr_y) + ctr_x)});
    float min_x= min( {(cos(rad) *(0-ctr_x) - sin(rad) * (0-ctr_y) + ctr_x) , (cos(rad) *((image.rows-1)-ctr_x) - sin(rad) * ((image.cols-1)-ctr_y) + ctr_x)  ,(cos(rad) *(0-ctr_x) - sin(rad) * ((image.cols-1)-ctr_y) + ctr_x) ,(cos(rad) *((image.rows-1)-ctr_x) - sin(rad) * (0-ctr_y) + ctr_x)});
    float max_y = max({(sin(rad) *(0-ctr_x) + cos(rad) * (0-ctr_y) + ctr_y) , (sin(rad) *((image.rows-1)-ctr_x) + cos(rad) * ((image.cols-1)-ctr_y) + ctr_y)  ,(sin(rad) *(0-ctr_x) + cos(rad) * ((image.cols-1)-ctr_y) + ctr_y), (sin(rad) *((image.rows-1)-ctr_x) + cos(rad) * (0-ctr_y) + ctr_y)}) ;
    float min_y= min({(sin(rad) *(0-ctr_x) + cos(rad) * (0-ctr_y) + ctr_y) , (sin(rad) *((image.rows-1)-ctr_x) + cos(rad) * ((image.cols-1)-ctr_y) + ctr_y)  ,(sin(rad) *(0-ctr_x) + cos(rad) * ((image.cols-1)-ctr_y) + ctr_y), (sin(rad) *((image.rows-1)-ctr_x) + cos(rad) * (0-ctr_y) + ctr_y)});
    
    Mat res = Mat::zeros(int(max_x - min_x+1), int(max_y-min_y+1),CV_32F);

    for(int x=0;x<tmp.rows;x++){
        for(int y=0;y<tmp.cols;y++){
            nv_x = cos(rad) *(x-ctr_x) - sin(rad) * (y-ctr_y) + ctr_x;
            nv_y = sin(rad) *(x-ctr_x) + cos(rad) * (y-ctr_y) + ctr_y;

            res.at<float>(nv_x -min_x,nv_y-min_y) = interpolationFunction(tmp,x,y);
        }
    }


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
    // res = expand(img,2,interpolate_nearest);
    // res.convertTo(res,CV_8UC1); 


    ////QST  3
    // res = expand(img,2,interpolate_bilinear);
    // res.convertTo(res,CV_8UC1); 
    
    ////QST  5
    // res = rotation(img,35.2,interpolate_nearest);
    // res.convertTo(res,CV_8UC1); 

    imshow("image22",res);
    waitKey(0);
    return 0;
}
