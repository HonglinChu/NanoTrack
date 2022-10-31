#include <iostream>
#include <cstdlib>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include "nanotrack.hpp" 

void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect &rect) 
{   
    rect.x = max(0, pos.x - int(sz.x / 2));
    rect.y = max(0, pos.y - int(sz.y / 2));
    rect.width = int(sz.x);   
    rect.height = int(sz.y);    
}

void track(NanoTrack *siam_tracker, const char *video_path)

{
    // Read video 
    cv::VideoCapture capture; 
    bool ret;
    if (strlen(video_path)==1)
        ret = capture.open(atoi(video_path));  
    else
        ret = capture.open(video_path); 

    // Exit if video not opened.
    if (!ret) 
        std::cout << "Open cap failed!" << std::endl;

    // Read first frame. 
    cv::Mat frame; 
    
    bool ok = capture.read(frame);
    if (!ok)
    {
        std::cout<< "Cannot read video file" << std::endl;
        return; 
    }
    
    // Select a rect.
    cv::namedWindow("demo"); 
    cv::Rect trackWindow = cv::selectROI("demo", frame); // 手动选择
    //cv::Rect trackWindow =cv::Rect(244,161,74,70);         // 固定值 
    
    // Initialize tracker with first frame and rect.
    State state; 
    
    //   frame, (cx,cy) , (w,h), state
    siam_tracker->init(frame, trackWindow);
    std::cout << "==========================" << std::endl;
    std::cout << "Init done!" << std::endl; 
    std::cout << std::endl; 
    cv::Mat init_window;
    frame(trackWindow).copyTo(init_window); 

    for (;;)
    {
        // Read a new frame.
        capture >> frame;
        if (frame.empty())
            break;

        // Start timer
        double t = (double)cv::getTickCount();
        // Update tracker 
        siam_tracker->track(frame); 
        // Calculate Frames per second (FPS)
        double fps = cv::getTickFrequency() / ((double)cv::getTickCount() - t);
        
        // Result to rect. 
        cv::Rect rect;
        cxy_wh_2_rect(siam_tracker->state.target_pos, siam_tracker->state.target_sz, rect);

        // Boundary judgment.
        cv::Mat track_window;
        if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= frame.cols && 0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= frame.rows)
        {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
        }

        // Display FPS 
        std::cout << "FPS: " << fps << std::endl;
        std::cout << std::endl;

        // Display result. 
        cv::imshow("demo", frame);
        cv::waitKey(20);  

        // Exit if 'q' pressed.
        if (cv::waitKey(20) == 'q')
        {
            break;
        }
    }
    cv::destroyWindow("demo");
    capture.release();
}

int main(int argc, char** argv)
{
    // if (argc != 2)
    // {
    //     fprintf(stderr, "Usage: %s [videopath(file or camera)]\n", argv[0]);
    //     return -1;
    // } 

    // Get model path.
    std::string backbone_model = "./model/ncnn/nanotrack_backbone_sim-opt";
    std::string head_model = "./model/ncnn/nanotrack_head_sim-opt"; // 

    // Get video path                    
    //const char* video_path = argv[1];
    const char* video_path = "./data/videos/girl_dance.mp4"; 
    
    // Build tracker 
    NanoTrack *siam_tracker; 
    
    // 动态创建的时候主要是加载了模型的 param 和 bin 参数  
    siam_tracker = new NanoTrack(); 
    siam_tracker->load_model(backbone_model, head_model); 
    track(siam_tracker, video_path); 

    return 0;
}
