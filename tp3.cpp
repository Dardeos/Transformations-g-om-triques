
#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <numbers>


using namespace cv;
using namespace std;


cv::Mat meanFilter(cv::Mat image, int k){
    
    k = 2*k+1;
    int pad = k/2;
    int sum = 0;
    Mat filt = Mat::ones(k,k,CV_8UC1) / (k*k);
    Mat res = Mat::zeros(image.rows+2*pad,image.cols+2*pad,CV_8UC1);   
    Mat tmp = res.clone(); 

    for(int x = 0;x<image.rows;x++){
        for (int y=0;y<image.cols;y++){
            tmp.at<uchar>(x + pad, y + pad) = image.at<uchar>(x, y);
        }
    }

    for(int x = pad;x<tmp.rows-pad;x++){
        for (int y=pad;y<tmp.cols-pad;y++){
            sum = 0;
            for(int i = 0;i<filt.rows;i++){
                for (int j=0;j<filt.cols;j++){
                    sum += tmp.at<uchar>(x-pad+i, y-pad+j) * filt.at<uchar>(i, j);
                }
            }
            res.at<uchar>(x,y) = sum;
        }
    }   

    return res;

}

Mat convolution(Mat image, cv::Mat kernel)
{
    image.convertTo(image, CV_32F);
    kernel.convertTo(kernel, CV_32F);
    int k = kernel.rows;
    int pad = k/2;
    float sum = 0;
    Mat res = image.clone();
    Mat tmp = Mat::zeros(image.rows+2*pad,image.cols+2*pad,CV_32F);   
    cout<<k;

    cout<<image.rows<<endl;
    for(int x = 0;x<image.rows;x++){
        for (int y=0;y<image.cols;y++){
            tmp.at<float>(x + pad, y + pad) = image.at<float>(x, y);
        }
    }
    for(int x = pad;x<tmp.rows-pad;x++){
        for (int y=pad;y<tmp.cols-pad;y++){
            sum = 0;
            for(int i = 0;i<kernel.rows;i++){
                for (int j=0;j<kernel.cols;j++){
                    sum += tmp.at<float>(x-pad+i, y-pad+j) * kernel.at<float>(i, j);
                }
            }
            res.at<float>(x-pad,y-pad) = sum;
        }
    }  

    
    return res;
}

cv::Mat edgeSobel(cv::Mat image)
{
    Mat resx = image.clone();
    Mat resy = image.clone();
    Mat sobelx = (Mat_<float>(3,3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);
    Mat sobely = (Mat_<float>(3,3) <<
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1);

    resx = convolution(image,sobelx);
    resy = convolution(image,sobely);
    
    Mat res = Mat::zeros(image.size(), CV_32F);
    for(int i=0; i<resx.rows; i++){
        for(int j=0; j<resx.cols; j++){
            float gx = resx.at<float>(i,j);
            float gy = resy.at<float>(i,j);
            res.at<float>(i,j) = sqrt(gx*gx + gy*gy);
        }
    }
    res.convertTo(res,CV_8UC1);
    return res;
}



float gaussian(float x, float sigma2)
{
    return 1.0/(2*M_PI*sigma2)*exp(-x*x/(2*sigma2));
}


cv::Mat bilateralFilter(cv::Mat image, cv::Mat kernel, float sigma_r)
{
    image.convertTo(image, CV_32F);
    kernel.convertTo(kernel, CV_32F);
    int k = kernel.rows;
    int pad = k/2;
    float sum = 0,weight;
    Mat res = image.clone();
    Mat tmp = Mat::zeros(image.rows+2*pad,image.cols+2*pad,CV_32F);   
    cout<<k;

    cout<<image.rows<<endl;
    for(int x = 0;x<image.rows;x++){
        for (int y=0;y<image.cols;y++){
            tmp.at<float>(x + pad, y + pad) = image.at<float>(x, y);
        }
    }
    for(int x = pad;x<tmp.rows-pad;x++){
        for (int y=pad;y<tmp.cols-pad;y++){
            sum = 0;
            weight=0;
            float center = tmp.at<float>(x, y);

            for(int i = 0;i<kernel.rows;i++){
                for (int j=0;j<kernel.cols;j++){
                    float Gr = (1/(sqrt(2*M_PI)*sigma_r))*exp((-0.5*(tmp.at<float>(x-pad+i, y-pad+j) - center)*(tmp.at<float>(x-pad+i, y-pad+j) - center))/(sigma_r*sigma_r));
                    sum += tmp.at<float>(x-pad+i, y-pad+j) * kernel.at<float>(i, j) * Gr;
                    weight += kernel.at<float>(i, j) * Gr;
                }
            }
            res.at<float>(x-pad,y-pad) = sum / weight;
        }
    }  

    
    return res;
}


cv::Mat median(cv::Mat image, int k)
{
    Mat res = image.clone();
    k = 2*k+1;
    int pad = k/2;
    std::vector<int> tab;
    Mat tmp = Mat::zeros(image.rows+2*pad,image.cols+2*pad,CV_8UC1);   

    for(int x = 0;x<image.rows;x++){
        for (int y=0;y<image.cols;y++){
            tmp.at<uchar>(x + pad, y + pad) = image.at<uchar>(x, y);
        }
    }

    for(int x = pad;x<tmp.rows-pad;x++){
        for (int y=pad;y<tmp.cols-pad;y++){
            
            std::vector<int> tab;
            for(int i = 0; i < k; i++){
                for(int j = 0; j < k; j++){
                    tab.push_back((int)tmp.at<uchar>(x - pad + i, y - pad + j));
                }
            }

            sort(tab.begin(), tab.end());

            int med = tab[tab.size() / 2];

            res.at<uchar>(x - pad, y - pad) = (uchar)med;
        }
    }


    return res;
}

int main(){    

    string path = "C:\\Users\\user\\OneDrive\\Bureau\\AnUniv\\m1-p\\image\\ImageProcessingLab-main\\pic\\camera2.png";
    Mat img = imread(path,IMREAD_GRAYSCALE);
    Mat res = img.clone();
    imshow("image",img);
    waitKey(0);
    
    ////QST  1
    //res = meanFilter(img,5);
    
    ////QST  2
    // Mat krn = (Mat_<int>(3,3) <<
    //  0,  1,  0,
    //  1, -4,  1,
    //  0,  1,  0);
    // res = convolution(img,krn);
    // res.convertTo(res,CV_8UC1);
    
    ////QST  3
    //res = edgeSobel(img);
    
    ////QST4
    // Mat krn = (Mat_<float>(3,3) <<
    // 1, 2, 1,
    // 2, 4, 2,
    // 1, 2, 1);
    // krn /= 16.0f;
    // res = bilateralFilter(img,krn,1.0f);
    // res.convertTo(res,CV_8UC1);

    ////QST  5
    //res = median(img,3);
    

    imshow("image22",res);
    waitKey(0);
    return 0;
}
