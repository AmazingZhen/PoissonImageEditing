// Copyright (C) 2019 Amazing Zhen <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//#define EIGEN_USE_MKL_ALL

using namespace cv;

Mat poisson_image_editing(Mat src, Mat dst, Mat mask, const bool mixing = true);

