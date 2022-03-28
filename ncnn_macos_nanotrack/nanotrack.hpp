#ifndef NANOTRACK_H
#define NANOTRACK_H

#include <vector> 
#include <map>  
 
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 

#include "net.h" 
#include "layer_type.h" 

#define PI 3.1415926 

using namespace cv;

struct Config{ 
    
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.15;
    float window_influence = 0.475;
    float lr = 0.38;
    int exemplar_size=127;
    int instance_size=255;
    int total_stride=16;
    int score_size=16;
    float context_amount = 0.5;
};

struct State { 
    int im_h; 
    int im_w;  
    cv::Scalar channel_ave; 
    cv::Point target_pos; 
    cv::Point2f target_sz = {0.f, 0.f}; 
    float cls_score_max; 
};

class NanoTrack {

public: 
    
    NanoTrack();
    
    ~NanoTrack(); 

    void init(cv::Mat img, cv::Rect bbox);
    
    void update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, float scale_z, float &cls_score_max);
    
    void track(cv::Mat im);
    
    void load_model(std::string model_backbone, std::string model_head);

    ncnn::Net net_backbone, net_head; 

    ncnn::Mat zf, xf; 

    int stride=16;
    
    // state  dynamic
    State state;
    
    // config static
    Config cfg; 

    const float mean_vals[3] = { 0.485f*255.f, 0.456f*255.f, 0.406f*255.f };  
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

private: 
    void create_grids(); 
    void create_window();  
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
};

#endif 