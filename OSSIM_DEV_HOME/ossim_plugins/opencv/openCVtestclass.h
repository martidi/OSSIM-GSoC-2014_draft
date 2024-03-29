
#include <ossim/base/ossimObject.h>
#include <ossim/base/ossimDpt.h>
#include <ossim/base/ossimString.h>
#include <ossim/base/ossimTieMeasurementGeneratorInterface.h>
#include "ossimIvtGeomXform.h"

#include <opencv/cv.h>

#include <ctime>
#include <vector>
#include <iostream>

class openCVtestclass
{
public:
   openCVtestclass();
   openCVtestclass(ossimRefPtr<ossimImageData> master, ossimRefPtr<ossimImageData> slave); 
   bool execute();
   
   cv::Mat master_mat, slave_mat;
   cv::vector<cv::KeyPoint> keypoints1, keypoints2;
   vector<cv::DMatch > good_matches;
};



               
