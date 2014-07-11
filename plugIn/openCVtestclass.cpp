
#include <ossim/base/ossimString.h>
#include <ossim/base/ossimNotify.h>
#include <ossim/base/ossimTrace.h>
#include <ossim/base/ossimIrect.h>
#include <ossim/base/ossimRefPtr.h>
#include <ossim/base/ossimConstants.h>
#include <ossim/elevation/ossimElevManager.h>
#include <ossim/imaging/ossimImageData.h>
#include <ossim/imaging/ossimImageSource.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "openCVtestclass.h"

#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/legacy/legacy.hpp>
// Note: These are purposely commented out to indicate non-use.
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/nonfree/features2d.hpp>
// Note: These are purposely commented out to indicate non-use.
#include <vector>
#include <iostream>

openCVtestclass::openCVtestclass()
{
	
}

openCVtestclass::openCVtestclass(ossimRefPtr<ossimImageData> master, ossimRefPtr<ossimImageData> slave)
{
	//Create the OpenCV images
    master_mat.create(cv::Size(master->getWidth(), master->getHeight()), CV_8UC1);
    slave_mat.create(cv::Size(slave->getWidth(), slave->getHeight()), CV_8UC1);
	
	memcpy(master_mat.ptr(), (void*) master->getBuf(), master->getWidth()*master->getHeight());
	memcpy(slave_mat.ptr(), (void*) slave->getBuf(), slave->getWidth()*slave->getHeight());
}

void openCVtestclass::run()
{
   cv::namedWindow( "master_img", CV_WINDOW_NORMAL );
   cv::imshow("master_img", master_mat);
   
   cv::namedWindow( "slave_img", CV_WINDOW_NORMAL );
   cv::imshow("slave_img", slave_mat);
   
   cv::waitKey(0);
}

void openCVtestclass::TPgen()
{
   	//computing detector
	cv::OrbFeatureDetector detector(400);
	detector.detect(master_mat, keypoints1);
	detector.detect(slave_mat, keypoints2);
	
	// computing descriptors
	cv::BriefDescriptorExtractor extractor;
	cv::Mat descriptors1, descriptors2;
	extractor.compute(master_mat, keypoints1, descriptors1);
	extractor.compute(slave_mat, keypoints2, descriptors2);

	// matching descriptors
	cv::BFMatcher matcher(cv::NORM_L2);
	vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);	

	// computing the results lower than a threshold based on distances 
	double max_dist = 0; double min_dist = 100;

	for( int i = 0; i < descriptors1.rows; i++ )
	{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	
	// error computation
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::median, boost::accumulators::tag::variance> > acc_x;
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::median, boost::accumulators::tag::variance> > acc_y;

	vector<cv::DMatch > good_matches;

	for( int i = 0; i < descriptors1.rows; i++ )
	{
		if(matches[i].distance <= std::max(2*min_dist, 320.50))
		{ 
	        // parallax computation
			double px = keypoints1[i].pt.x - keypoints2[matches[i].trainIdx].pt.x;
			double py = keypoints1[i].pt.y - keypoints2[matches[i].trainIdx].pt.y;	
			
			if(fabs(px) <= 200 && fabs(py) <= 200)	
			{
				good_matches.push_back(matches[i]);

				acc_x(px);
				acc_y(py);
		         
		    	cout << i << " " << px << " "
		              << " " << py << " "
		              <<endl;
			}	
		}
	}
}

void openCVtestclass::TPdraw()
{
	//Drawing the results
	cv::Mat img_matches;
	cv::drawMatches(master_mat, keypoints1, slave_mat, keypoints2,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	cv::resize(img_matches, img_matches, cv::Size(), 1.0/8.0, 1.0/8.0, cv::INTER_AREA);

	cv::namedWindow("TP matched", CV_WINDOW_NORMAL );
	cv::imshow("TP matched", img_matches );	
   
	cv::waitKey(0);
}
