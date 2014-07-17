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
	// Create the OpenCV images
    master_mat.create(cv::Size(master->getWidth(), master->getHeight()), CV_8UC1);
    slave_mat.create(cv::Size(slave->getWidth(), slave->getHeight()), CV_8UC1);
	
	memcpy(master_mat.ptr(), (void*) master->getBuf(), master->getWidth()*master->getHeight());
	memcpy(slave_mat.ptr(), (void*) slave->getBuf(), slave->getWidth()*slave->getHeight());
	
	cv::transpose(master_mat, master_mat);
	cv::flip(master_mat, master_mat, 1);
	
	cv::transpose(slave_mat, slave_mat);
	cv::flip(slave_mat, slave_mat, 1);
}

void openCVtestclass::disparity()
{
	/// Disparity Map generation
	cv::Mat imgDisparity16S = cv::Mat( master_mat.rows, master_mat.cols, CV_16S );
	cv::Mat imgDisparity8U  = cv::Mat( master_mat.rows, master_mat.cols, CV_8UC1 );

	int ndisparities = 16*2;   
	int SADWindowSize = 3;   

	cv::StereoSGBM sgbm;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	int cn = master_mat.channels();

	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = -10;
	sgbm.numberOfDisparities = ndisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	
	double minVal, maxVal;

	/// images rotation for SGM computation in every directions
	cv::Mat array_disp[4];
    cv::Mat array_disp_8U[4];

	for(int i=0; i<=3; i++)
	{	
		cv::transpose(master_mat, master_mat);
		cv::flip(master_mat, master_mat, 1);

		transpose(slave_mat, slave_mat);
		flip(slave_mat, slave_mat, 1);

		//Mat disp, disp8U;
		sgbm(master_mat, slave_mat, array_disp[i]);
		minMaxLoc( array_disp[i], &minVal, &maxVal );
		array_disp[i].convertTo( array_disp_8U[i], CV_8UC1, 255/(maxVal - minVal), -minVal*255/(maxVal - minVal));
	}
    
	cv::namedWindow( "SGM Disparity1", CV_WINDOW_NORMAL );
	cv::imshow( "SGM Disparity1", array_disp_8U[3] );

	cv::namedWindow( "SGM Disparity2", CV_WINDOW_NORMAL );
	cv::imshow( "SGM Disparity2", array_disp_8U[0] );

	cv::namedWindow( "SGM Disparity3", CV_WINDOW_NORMAL );
	cv::imshow( "SGM Disparity3", array_disp_8U[1] );

	cv::namedWindow( "SGM Disparity4", CV_WINDOW_NORMAL );
	cv::imshow( "SGM Disparity4", array_disp_8U[2] );		

	cv::waitKey(0);
}

cv::Mat openCVtestclass::estRT(std::vector<cv::Point2f> master, std::vector<cv::Point2f> slave)
{
	size_t m = master.size();
	
	if ( master.size() != slave.size() ) 
	  {
         throw 0;
      }
    cout << m << endl;
    
    // computing barycentric coordinates
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::median> > acc_x_master;
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::median> > acc_y_master;
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::median> > acc_x_slave;
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::median> > acc_y_slave;

	for( int i = 0; i < m; i++)
	{
		acc_x_master(master[i].x);
		acc_y_master(master[i].y);
		acc_x_slave(slave[i].x);
		acc_y_slave(slave[i].y);
	}

	double master_x = boost::accumulators::median(acc_x_master);
    double master_y = boost::accumulators::median(acc_y_master);
	double slave_x = boost::accumulators::median(acc_x_slave);
	double slave_y = boost::accumulators::median(acc_y_slave);

		cout << "median_x_master = " << master_x << endl
			 << "median_y_master = " << master_y << endl
			 << "median_x_slave = "  << slave_x  << endl
			 << "median_y_slave = "  << slave_y  << endl; 

	std::vector<cv::Point2f> bar_master, bar_slave;

	for (int i = 0; i < m; i++)
    {
        cv::Point2f pt1;
        cv::Point2f pt2;

        pt1.x = master[i].x - master_x;
        pt1.y = master[i].y - master_y;

        pt2.x = slave[i].x - slave_x;
        pt2.y = slave[i].y - slave_y;

        bar_master.push_back(pt1);
        bar_slave.push_back(pt2);
    }

    cv::Mat x_approx = cv::Mat::zeros (3,1,6);
    cv::Mat result = cv::Mat::zeros (3, 1, 6);
    cv::Mat A = cv::Mat::zeros(m,3,6);
    cv::Mat B = cv::Mat::zeros(m,1,6);

    for (int j= 0; j <10; j++)
    {
    for (int i=0; i < m ; i++)
		{
			A.at<double>(i,0) = -bar_slave[i].x;
			A.at<double>(i,1) = 0.0;
			A.at<double>(i,2) = 1.0;

			B.at<double>(i,0) = bar_master[i].y + sin(x_approx.at<double>(0,0))*bar_slave[i].x - 
							cos(x_approx.at<double>(0,0))*bar_slave[i].y - x_approx.at<double>(2,0); 
		 }

	cv::solve(A, B, result, cv::DECOMP_SVD);
	x_approx = x_approx+result;

	cout << "matrice risultato" << endl;
	cout << x_approx << endl;
	}
	
	cv::Point2f pt(master_mat.rows/2 , master_mat.cols/2);
	// rotation is applied in the barycenter
    cv::Mat r = getRotationMatrix2D(pt, -x_approx.at<double>(0,0)*180.0/3.141516, 1.0);

    r.at<double>(1,2) += master_y - slave_y;
	
	return r;
}

void openCVtestclass::run()
{
   cv::namedWindow( "master_img", CV_WINDOW_NORMAL );
   cv::imshow("master_img", master_mat);
   
   cv::namedWindow( "slave_img", CV_WINDOW_NORMAL );
   cv::imshow("slave_img", slave_mat);
   
   //cv::waitKey(0);
}

void openCVtestclass::TPdraw()
{
	// Drawing the results
	cv::Mat img_matches;
	cv::drawMatches(master_mat, keypoints1, slave_mat, keypoints2,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	cv::resize(img_matches, img_matches, cv::Size(), 1.0/2.0, 1.0/2.0, cv::INTER_AREA);

	cv::namedWindow("TP matched", CV_WINDOW_NORMAL );
	cv::imshow("TP matched", img_matches );	
   
	// cv::waitKey(0);
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

void openCVtestclass::warp()
{
	std::vector<cv::Point2f> aff_match1, aff_match2;
    for (int i = 0; i < good_matches.size(); ++i)
    {	/// get the keypoints from the good_matches
        cv::Point2f pt1 = keypoints1[good_matches[i].queryIdx].pt;
        cv::Point2f pt2 = keypoints2[good_matches[i].trainIdx].pt;
        aff_match1.push_back(pt1);
        aff_match2.push_back(pt2);
        printf("%3d pt1: (%.2f, %.2f) pt2: (%.2f, %.2f)\n", i, pt1.x, pt1.y, pt2.x, pt2.y);
    }

	cv::Mat rot_matrix = estRT(aff_match2, aff_match1);
    cout << rot_matrix << endl;
    /// Set the destination image the same type and size as source
	cv::Mat warp_dst = cv::Mat::zeros(master_mat.rows, master_mat.cols, master_mat.type());
	cout << warp_dst.size () << endl;
    cv::warpAffine(slave_mat, warp_dst, rot_matrix, warp_dst.size());

	cv::namedWindow("Master image", CV_WINDOW_NORMAL);
	cv::imshow("Master image", master_mat );

	cv::namedWindow("Warped image", CV_WINDOW_NORMAL);
	cv::imshow("Warped image", warp_dst );

    cv::waitKey(0);
}	

