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
#include "TPgenerator.h"
#include "DisparityMap.h"

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
    master_mat.create(cv::Size(master->getWidth(), master->getHeight()), CV_16UC1);
    slave_mat.create(cv::Size(slave->getWidth(), slave->getHeight()), CV_16UC1);
	
	memcpy(master_mat.ptr(), (void*) master->getUshortBuf(), 2*master->getWidth()*master->getHeight());
	memcpy(slave_mat.ptr(), (void*) slave->getUshortBuf(), 2*slave->getWidth()*slave->getHeight());
	
	cout << "coversione effettuata" << endl;
	
	// Rotation for along-track images
	cv::transpose(master_mat, master_mat);
	cv::flip(master_mat, master_mat, 1);
	
	cv::transpose(slave_mat, slave_mat);
	cv::flip(slave_mat, slave_mat, 1);
}

bool openCVtestclass::execute()
{
		double minVal_master, maxVal_master, minVal_slave, maxVal_slave;
		cv::Mat master_mat_8U;
		cv::Mat slave_mat_8U;  
      
   		minMaxLoc( master_mat, &minVal_master, &maxVal_master );
   		minMaxLoc( slave_mat, &minVal_slave, &maxVal_slave );
		master_mat.convertTo( master_mat_8U, CV_8UC1, 255.0/(maxVal_master - minVal_master), -minVal_master*255.0/(maxVal_master - minVal_master));
		slave_mat.convertTo( slave_mat_8U, CV_8UC1, 255.0/(maxVal_slave - minVal_slave), -minVal_slave*255.0/(maxVal_slave - minVal_slave)); 
	
		TPgenerator* TPfinder = new TPgenerator(master_mat_8U, slave_mat_8U);
		TPfinder->run();
		TPfinder->TPgen();
		TPfinder->TPdraw();
		//TPfinder->warp();
	
	
	cv::Mat slave_mat_warp = TPfinder->warp(slave_mat);
	
	//cv::Ptr<cv::CLAHE> filtro = cv::createCLAHE();
    //filtro->apply(master_mat_8U, master_mat_8U); 
    //filtro->apply(slave_mat_warp, slave_mat_warp);
    cv::imwrite("Master_8bit_bSGM.tif",  master_mat_8U);
    cv::imwrite("Slave_8bit_bSGM.tif",  slave_mat_warp);
    	
	DisparityMap* dense_matcher = new DisparityMap();
	dense_matcher->execute(master_mat_8U, slave_mat_warp);
	
	return true;
	
}



