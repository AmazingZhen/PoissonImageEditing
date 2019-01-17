#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Poisson.h"

const char src_path[] = "input/src.jpg";
const char dst_path[] = "input/dst.jpg";
const char mask_path[] = "input/mask.png";

using namespace cv;
using namespace std;

Mat src, dst, mask;

void create_mask();

int main()
{
	mask = imread(mask_path, CV_8UC1);
	if (mask.empty()) {
		create_mask();
		return 0;
	}

	src = imread(src_path);
	dst = imread(dst_path);

	if (src.empty() || dst.empty() || src.size() != dst.size() || src.size() != mask.size())
	{
		return -1;
	}

	Mat seamless_cloning_res;
	seamless_cloning_res = poisson_image_editing(src, dst, mask, false);
	imwrite("res/seamless_cloning_res.jpg", seamless_cloning_res);

	Mat mixed_seamless_cloning_res;
	mixed_seamless_cloning_res = poisson_image_editing(src, dst, mask, true);
	imwrite("res/mixed_seamless_cloning_res.jpg", mixed_seamless_cloning_res);

	getchar();
	return 0;
}

bool mousedown;
vector<vector<Point> > contours;
vector<Point> pts;

void onMouse(int event, int x, int y, int flags, void* userdata)
{
	Mat clone_src = *((Mat *)userdata);

	if (event == EVENT_LBUTTONDOWN)
	{
		mousedown = true;
		contours.clear();
		pts.clear();
	}

	if (event == EVENT_LBUTTONUP)
	{
		mousedown = false;
		if (pts.size() > 2)
		{
			mask = 0;
			contours.push_back(pts);
			drawContours(mask, contours, 0, Scalar(255), -1);

			Mat masked(src.size(), CV_8UC3, Scalar(255, 255, 255));
			src.copyTo(masked, mask);
			imshow("Masked src", masked);
			imwrite(mask_path, mask);
		}
	}

	if (mousedown)
	{
		if (pts.size() > 2)
			line(clone_src, Point(x, y), pts[pts.size() - 1], Scalar(0, 255, 0));

		pts.push_back(Point(x, y));

		imshow("Create Mask", clone_src);
	}
}

void create_mask() {
	src = imread(src_path);
	dst = imread(dst_path);

	if (src.empty() || dst.empty() || src.size() != dst.size())
	{
		return;
	}

	mask = Mat(src.size(), CV_8UC1);
	mask = 0;

	namedWindow("Create Mask", WINDOW_AUTOSIZE);
	Mat clone_src = src.clone();
	setMouseCallback("Create Mask", onMouse, &clone_src);
	imshow("Create Mask", src);

	waitKey(0);
}