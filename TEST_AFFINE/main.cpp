#include <stdio.h>
#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>

using namespace boost::accumulators;
using namespace cv;
using namespace std;

//const char *windowDisparitySGM1 = "SGM Disparity1";

const char *windowMatch = "TP matched";
const char *source_window = "Master image";
const char *warp_window = "Warped image";
//char *warp_window = "Warp";
//char *warp_rotate_window = "Warp + Rotate";

Mat estRT (vector<Point2f> master, vector<Point2f> slave);

int main(int argc, const char* argv[])
{
	cout <<"File Left:  "<< argv[1] <<"\n";
	cout <<"File Right: "<< argv[2] <<"\n";

	Mat rotated_Left =  imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	Mat rotated_Right = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
	
	if( !rotated_Left.data || !rotated_Right.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }
	
	//resampling
	Mat outputLeft, outputRight;
	resize(rotated_Left, outputLeft, Size(), 1.0/8.0, 1.0/8.0, INTER_AREA);
	resize(rotated_Right, outputRight, Size(), 1.0/8.0, 1.0/8.0, INTER_AREA);


	//computing detector
	OrbFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(outputLeft, keypoints1);
	detector.detect(outputRight, keypoints2);
	
	// computing descriptors
	BriefDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(outputLeft, keypoints1, descriptors1);
	extractor.compute(outputRight, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// drawing the results lower than a threshold based on distances 
	double max_dist = 0; double min_dist = 100;

	for( int i = 0; i < descriptors1.rows; i++ )
	{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	cout <<"Max dist:  "<< max_dist <<"\n";
	cout <<"Min dist: "<< min_dist <<"\n";

	// error computation
	boost::accumulators::accumulator_set<double, stats<tag::mean, tag::median, tag::variance> > acc_x;

	boost::accumulators::accumulator_set<double, stats<tag::mean, tag::median, tag::variance> > acc_y;

	vector< DMatch > good_matches;

	for( int i = 0; i < descriptors1.rows; i++ )
	{
		if(matches[i].distance <= std::max(2*min_dist, 250.50))
		{ 
	        // parallax computation
			double px = keypoints1[i].pt.x - keypoints2[matches[i].trainIdx].pt.x;
			double py = keypoints1[i].pt.y - keypoints2[matches[i].trainIdx].pt.y;	
			
			if(fabs(px) <= 200 && fabs(py) <= 200)	
			{
				good_matches.push_back( matches[i]);

				acc_x(px);
				acc_y(py);
		         
		    	cout << i << " " << px << " "
		              << " " << py << " "
		              <<endl;
			}
			
		    }
	}

	cout << "mean_x = " << boost::accumulators::mean(acc_x) 		<< endl
	 << "mean_y = " 	<< boost::accumulators::mean(acc_y) 		<< endl
	 << "median_x = " 	<< boost::accumulators::median(acc_x) 		<< endl
	 << "median_y = " 	<< boost::accumulators::median(acc_y)		<< endl
	 << "st.dev_x = " 	<< sqrt(boost::accumulators::variance(acc_x)) << endl
	 << "st.dev_y = " 	<< sqrt(boost::accumulators::variance(acc_y)) << endl;

	int delta_y = boost::accumulators::median(acc_y);
 	

	Mat img_matches;
	drawMatches( outputLeft, keypoints1, outputRight, keypoints2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
    namedWindow( windowMatch, CV_WINDOW_NORMAL );
	imshow( windowMatch, img_matches );	

	//PROVA WARP
	vector<Point2f> aff_match1, aff_match2;
    for (int i = 0; i < good_matches.size(); ++i)
    {
        Point2f pt1 = keypoints1[good_matches[i].queryIdx].pt;
        Point2f pt2 = keypoints2[good_matches[i].trainIdx].pt;
        aff_match1.push_back(pt1);
        aff_match2.push_back(pt2);
        printf("%3d pt1: (%.2f, %.2f) pt2: (%.2f, %.2f)\n", i, pt1.x, pt1.y, pt2.x, pt2.y);
    }

    Mat warp_mat = estimateRigidTransform(aff_match2, aff_match1, false);

    Mat warp_new = estRT(aff_match2, aff_match1);

    cout << warp_mat <<endl;

    // COMPUTE BORDER SIZE

	int rows = rotated_Left.rows;
	int cols = rotated_Left.cols;

	Size s = rotated_Left.size();
	rows = s.height;
	cols = s.width;

    cout << rows << " " << cols << endl;

    //cout << warp_mat.type() << endl;
    //cout << A22 << endl;
	Mat A = warp_mat(Rect(0,0,2,2));
	Mat B = warp_mat.col(2);
	cout << A << "" << B << endl;
	Mat pix = Mat::zeros(2,1, A.type());
	pix.at<double>(0,0) = 0.0;
	pix.at<double>(1,0) = 0.0;
	Mat ris = A*pix+B;
    cout << ris << endl;

	pix.at<double>(0,0) = 0.0;
	pix.at<double>(1,0) = cols;
	ris = A*pix+B;
    cout << ris << endl;

 	pix.at<double>(0,0) = rows;
	pix.at<double>(1,0) = 0.0;
	ris = A*pix+B;
    cout << ris << endl;   

    pix.at<double>(0,0) = rows;
	pix.at<double>(1,0) = cols;
	ris = A*pix+B;
    cout << ris << endl;

   /// Set the dst image the same type and size as src
   Mat warp_dst = Mat::zeros( outputLeft.rows, outputLeft.cols, outputRight.type() );

	// Apply the Homography Transform
   //warpPerspective( outputRight, warp_dst, warp_mat, warp_dst.size() );
   /// Apply the Affine Transform just found to the src image
   warpAffine( outputRight, warp_dst, warp_mat, warp_dst.size() );

   /// Show what you got
   namedWindow( source_window, CV_WINDOW_NORMAL );
   imshow( source_window, outputLeft );

   namedWindow( warp_window, CV_WINDOW_NORMAL );
   imshow( warp_window, warp_dst );

    waitKey(0);

	return 0;
}

//FUNZIONE CON CUI STIMO LA ROTOTRASLAZIONE
Mat estRT (vector<Point2f> master, vector<Point2f> slave)
{
    size_t m = master.size();

	if ( master.size() != slave.size() ) 
	  {
         throw 0;
      }

    cout << m << endl;

    //computing barycentric coordinates
	boost::accumulators::accumulator_set<double, stats<tag::mean> > acc_x_master;

	boost::accumulators::accumulator_set<double, stats<tag::mean> > acc_y_master;

	boost::accumulators::accumulator_set<double, stats<tag::mean> > acc_x_slave;

	boost::accumulators::accumulator_set<double, stats<tag::mean> > acc_y_slave;

	for( int i = 0; i < m; i++ )
	{
		acc_x_master(master[i].x);
		acc_y_master(master[i].y);
		acc_x_slave(slave[i].x);
		acc_y_slave(slave[i].y);
	}

	double master_x = boost::accumulators::mean(acc_x_master);
    double master_y = boost::accumulators::mean(acc_y_master);
	double slave_x = boost::accumulators::mean(acc_x_slave);
	double slave_y = boost::accumulators::mean(acc_y_slave);

	

	cout << "mean_x_master = " << boost::accumulators::mean(acc_x_master) 	<< endl
		 << "mean_y_master = " << boost::accumulators::mean(acc_y_master) 	<< endl
		 << "mean_x_slave = "  << boost::accumulators::mean(acc_x_slave) 	<< endl
		 << "mean_y_slave = "  << boost::accumulators::mean(acc_y_slave) 	<< endl; 

	vector<Point2f> bar_master, bar_slave;

	for (int i = 0; i < m; ++i)
    {
        Point2f pt1;
        Point2f pt2;

        pt1.x = master[i].x - master_x;
        pt1.y = master[i].y - master_y;

        pt2.x = slave[i].x - slave_x;
        pt2.y = slave[i].y - slave_y;

        bar_master.push_back(pt1);
        bar_slave.push_back(pt2);
    }

    Mat A = Mat::zeros(2*m,3,6);

    Mat B = Mat::zeros(2*m,1,6);

    Mat x_approx = Mat::zeros (3,1,6);

    Mat result = Mat::zeros (3, 1, 6);

    for (int j= 0; j <10; j++)
    {

    for (int i=0; i < 2*m ; i=i+2)
    {

    	A.at<double>(i,0) = bar_slave[i].y;
		A.at<double>(i,1) = 1.0;
		A.at<double>(i,2) = 0.0;	

		A.at<double>(i+1,0) = -bar_slave[i].x;
		A.at<double>(i+1,1) = 0.0;
		A.at<double>(i+1,2) = 1.0;

		B.at<double>(i,0) = bar_master[i].x -cos(x_approx.at<double>(0,0))*bar_slave[i].x - 
							sin(x_approx.at<double>(0,0))*bar_slave[i].y - x_approx.at<double>(1,0); 

		B.at<double>(i+1,0) = bar_master[i].y + sin(x_approx.at<double>(0,0))*bar_slave[i].x - 
							cos(x_approx.at<double>(0,0))*bar_slave[i].y - x_approx.at<double>(2,0); 
    }

    //cout << B << endl;
	Mat N = A.t()*A;
	cout << "matrice N" << endl;
	cout << N << endl;
	Mat N_1 = N.inv(DECOMP_SVD);
	cout << N_1 << endl;
	Mat risultato = N_1*A.t()*B;
	//cout << "matrice risultato" << endl;
	cout << risultato << endl;

	solve(A, B, result, DECOMP_SVD);

	x_approx = x_approx+result;

	cout << "matrice risultato" << endl;
	cout << result << endl;

	}

	return result;
}