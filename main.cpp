#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<string>
#include<thread>
#include <mutex> 
#include<chrono>
#include<omp.h>

using namespace cv;
using namespace std;

#define WIDTH 5120            //要处理图像的宽高
#define HEIGHT 5120
#define M 8                   //处理后图像数量M*N
#define N 8
#define SUB_WIDTH WIDTH/ M    //分割后图像的宽高 8192/4=2048 即为2048*2048可根据需要调整
#define SUB_HEIGHT HEIGHT/ N  //例如 M=8 N=8 分为1024*1024大小
const int OVERLAP = 100;
mutex mtx;

/*
* >4000*4000  ->8*8
* 3000*3000<4000*4000 ->6*6
* 2000*2000<3000*3000  ->4*4
* 1000*1000<2000*20000  ->2*2
* <1000*1000            ->1*1

*/



struct CeilPostion
{
	Mat ceil_image;
	int x;
	int y;

};

vector<CeilPostion>ceil_postion;


vector<CeilPostion> split_image(Mat src) {

	vector<CeilPostion>ceil_postion;
	// resize(src, src, cv::Size(WIDTH, HEIGHT));


	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			if (i == 0 && j == 0) {

				Mat image_cut, roi_img;
				Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT, SUB_WIDTH+ OVERLAP, SUB_HEIGHT+ OVERLAP);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();

				ceil_postion.push_back({ roi_img,i * SUB_WIDTH,j * SUB_HEIGHT });
			}
			else if (i == 0 && j != 0) {
				Mat image_cut, roi_img;
				Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT-OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT+ OVERLAP);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				ceil_postion.push_back({ roi_img,i * SUB_WIDTH, j * SUB_HEIGHT - OVERLAP });
			}
			else if (i != 0 && j == 0) {
				Mat image_cut, roi_img;
				Rect rect(i * SUB_WIDTH-OVERLAP, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				ceil_postion.push_back({ roi_img,i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT });
			}
			else {
				Mat image_cut, roi_img;
				Rect rect(i * SUB_WIDTH-OVERLAP, j * SUB_HEIGHT-OVERLAP, SUB_WIDTH+OVERLAP, SUB_HEIGHT+OVERLAP);
				image_cut = Mat(src, rect);
				roi_img = image_cut.clone();
				ceil_postion.push_back({ roi_img,i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT - OVERLAP });
			}
		}
	}
	return ceil_postion;
}

Mat merge_image(vector<CeilPostion>ceil_img) {
	int t = 0;
	Mat MergeImage(Size(WIDTH, HEIGHT), CV_8UC3);
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{

			if (i == 0 && j == 0) {
				Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
				ceil_img[t].ceil_image.copyTo(MergeImage(rect));
			}
			else if (i == 0 && j != 0) {
				Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT - OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
				ceil_img[t].ceil_image.copyTo(MergeImage(rect));
			}
			else if (i != 0 && j == 0) {
				Rect rect(i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
				ceil_img[t].ceil_image.copyTo(MergeImage(rect));
			}
			else {
				Rect rect(i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT - OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
				ceil_img[t].ceil_image.copyTo(MergeImage(rect));
			}
			t++;
		}
	}
	//cv::imwrite("test.png", MergeImage);
	return MergeImage;
}



/*

int main() {
	unsigned int nCpu = std::max(std::thread::hardware_concurrency(), (unsigned int)1);
	cout << nCpu << endl;
	Mat src = imread("0.png");
	cv::resize(src, src, cv::Size(WIDTH, HEIGHT));
	Mat test = imread("test.png");
	Mat diff = src - test;
	vector<CeilPostion> ceil_img;
	auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++) {
		split_image(src);
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
	std::cout << "total_time=" << totalTime << std::endl;

	ceil_postion.size();
	merge_image(ceil_img);

	//图像合并
	return 0;
}
*/

/*
#include <opencv2/highgui/highgui.hpp>// 提供imread读取图片函数
using namespace cv;

#include <iostream>
#include <thread> // 提供线程类
#include <mutex> // 提供互斥锁类(对部分和进行累加的时候需要加锁)，很好用

using namespace std;

mutex mtx;// 定义一个互斥锁
long totalSum;// 总和
const enum RangeSpecify { LEFT_UP, LEFT_DOWN, RIGHT_UP, RIGHT_DOWN };
vector<Mat>ceil_image_vector(0);
void ImageAverage(Mat& img, enum RangeSpecify r)// 线程代码
{
	int startRow, startCol, endRow, endCol;
	switch (r) {
	case LEFT_UP:
		startRow = 0;
		endRow = img.rows / 2;
		startCol = 0;
		endCol = img.cols / 2;
		break;
	case LEFT_DOWN:
		startRow = img.rows / 2;
		endRow = img.rows;
		startCol = 0;
		endCol = img.cols / 2;
		break;
	case RIGHT_UP:
		startRow = 0;
		endRow = img.rows / 2;
		startCol = img.cols / 2;
		endCol = img.cols;
		break;
	case RIGHT_DOWN:
		startRow = img.rows / 2;
		endRow = img.rows;
		startCol = img.cols / 2;
		endCol = img.cols;
		break;
	}
	double t = (double)getTickCount();

	Mat roi;
	img({ startCol,startRow,endCol - startCol,endRow - startRow }).copyTo(roi);
	mtx.lock();// 在访问公共变量totalSum 之前对其进行加锁
	//totalSum += sum;
	ceil_image_vector.push_back(roi);
	mtx.unlock();// 访问完毕立刻解锁

	cout << r << " : " << sum << endl;
	cout << "task completed! Time elapsed " << (double)getTickCount() - t << endl;// 打印本次线程时间花费
}

int main()
{
	Mat src = imread("0.png", CV_LOAD_IMAGE_GRAYSCALE);


	double t = (double)getTickCount();
	thread t0(ImageAverage, ref(src), LEFT_UP);
	thread t1(ImageAverage, ref(src), LEFT_DOWN);
	thread t2(ImageAverage, ref(src), RIGHT_UP);
	thread t3(ImageAverage, ref(src), RIGHT_DOWN);

	t0.join();// 等待子线程t0执行完毕
	t1.join();
	t2.join();
	t3.join();

	for (int i = 0; i < ceil_image_vector.size(); i++) {
		Mat show = ceil_image_vector[i];
		cout << " " << endl;
	}
	cout << endl << "多线程总时间花费：" << (double)getTickCount() - t << endl;
	cout << "图像均值(多线程）: " << totalSum * 1.0 / (src.cols * src.rows) << endl << endl;

	// 验证准确性
	long sum = 0;

	t = (double)getTickCount();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum += src.at<unsigned char>(i, j);
		}
	}

	cout << "参照时间花费：" << (double)getTickCount() - t << endl;
	cout << "参照均值： " << sum * 1.0 / (src.rows * src.cols) << endl << endl;

	system("pause");
	return 0;
}
*/






vector<CeilPostion> crop_image(Mat& src, int j, int i) {

	vector<CeilPostion>ceil_postion;
	if (i == 0 && j == 0) {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();
		mtx.lock();
		//#pragma omp critical
		//		{
		ceil_postion.push_back({ roi_img,i * SUB_WIDTH,j * SUB_HEIGHT });
		//		}
		mtx.unlock();

	}
	else if (i == 0 && j != 0) {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT - OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();
		mtx.lock();
		//#pragma omp critical
		//		{
		ceil_postion.push_back({ roi_img,i * SUB_WIDTH, j * SUB_HEIGHT - OVERLAP });
		//		}

		mtx.unlock();
	}
	else if (i != 0 && j == 0) {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();
		mtx.lock();
		//#pragma omp critical
		//		{
		ceil_postion.push_back({ roi_img,i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT });
		//		}
		mtx.unlock();
	}
	else {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT - OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();
		mtx.lock();
		//#pragma omp critical
		//		{
		ceil_postion.push_back({ roi_img,i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT - OVERLAP });
		//		}
		mtx.unlock();
	}

	return ceil_postion;

}

vector<CeilPostion>  crop_image_openmp(Mat& src, int j, int i) {

	vector<CeilPostion> ceil_postion;
	if (i == 0 && j == 0) {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();

#pragma omp critical
		{
			ceil_postion.push_back({ roi_img,i * SUB_WIDTH,j * SUB_HEIGHT });
		}

	}
	else if (i == 0 && j != 0) {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT - OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();

#pragma omp critical
		{
			ceil_postion.push_back({ roi_img,i * SUB_WIDTH, j * SUB_HEIGHT - OVERLAP });
		}


	}
	else if (i != 0 && j == 0) {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();

#pragma omp critical
		{
			ceil_postion.push_back({ roi_img,i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT });
		}

	}
	else {
		Mat image_cut, roi_img;
		Rect rect(i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT - OVERLAP, SUB_WIDTH + OVERLAP, SUB_HEIGHT + OVERLAP);
		image_cut = Mat(src, rect);
		roi_img = image_cut.clone();

#pragma omp critical
		{
			ceil_postion.push_back({ roi_img,i * SUB_WIDTH - OVERLAP, j * SUB_HEIGHT - OVERLAP });
		}

	}

	return ceil_postion;
}


void execute_crop_image(cv::Mat src) {
	vector<std::thread>thread_pool;
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			thread_pool.push_back(thread(crop_image, ref(src), j, i));
		}
	}
	for (vector<thread>::iterator it = thread_pool.begin(); it != thread_pool.end(); ++it)
	{
		it->join();
	}

}


void execute_crop_image_openmp(cv::Mat src) {
	int id = omp_get_thread_num(); //获得当前并行区域中活动线程个数
#pragma omp parallel for num_threads(8)
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			crop_image_openmp(src, j, i);
		}
	}

}
int main() {
	Mat src = imread("0.png");

	auto num_thread = thread::hardware_concurrency();
	if (M * N > (num_thread/2)) {
		split_image(src);
	}
	else {
		execute_crop_image(src);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++) {
		
		execute_crop_image(src);
	
		execute_crop_image_openmp(src);
		
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
	std::cout << "total_time=" << totalTime << std::endl;




	return 0;
}
