//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: ossim-opencv.cpp
//
// Author:  David Burken
//
// Description: Contains application definition "ossim-opencv" app.
//
// NOTE:  This is supplied for simple quick test. DO NOT checkin your test to
//        the svn repository.  Simply edit ossim-opencv.cpp and run your test.
//        After completion you can do a "svn revert foo.cpp" if you want to
//        keep your working repository up to snuff.
//
// $Id: ossim-opencv.cpp 20095 2011-09-14 14:37:26Z dburken $
//----------------------------------------------------------------------------
#include <ossim/base/ossimArgumentParser.h>
#include <ossim/base/ossimApplicationUsage.h>
#include <ossim/base/ossimConstants.h> 
#include <ossim/base/ossimException.h>
#include <ossim/base/ossimNotify.h>
#include <ossim/init/ossimInit.h>

// Put your includes here:
#include <ossim/base/ossimRefPtr.h>
#include <ossim/base/ossimTimer.h>
#include <ossim/base/ossimTrace.h>
#include <ossim/util/ossimChipperUtil.h>

#include "openCVtestclass.h"
#include "DisparityMap.h"
#include "ossimTieMeasurementGenerator.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>

#include <iostream>
#include <sstream>
#include <cstdlib> /* for exit */
#include <iomanip>

#include "ossim/imaging/ossimImageHandlerRegistry.h"
#include "ossim/imaging/ossimImageHandler.h"

using namespace std;

bool ortho (ossimArgumentParser argPars)
{
      // Make the generator.
      ossimRefPtr<ossimChipperUtil> chipper = new ossimChipperUtil;

      try
      	{      
         bool continue_after_init = chipper->initialize(argPars);
         if (continue_after_init)
         	{      
            // ossimChipperUtil::execute can throw an exception.
            chipper->execute();
            
            ossimNotify(ossimNotifyLevel_NOTICE)
               << "elapsed time in seconds: "
               << std::setiosflags(ios::fixed)
               << std::setprecision(3)
               << ossimTimer::instance()->time_s() << endl;
         	}
      	}
      catch (const ossimException& e)
     	{
         ossimNotify(ossimNotifyLevel_WARN) << e.what() << endl;
         exit(1);
     	}
   return true;
}


static ossimTrace traceDebug = ossimTrace("ossim-chipper:debug");

int main(int argc,  char *argv[])
{
   // Initialize ossim stuff, factories, plugin, etc.
   ossimTimer::instance()->setStartTick();
   	try
   		{ 
        char* argv_master[10];
        char* argv_slave[10];
        
        cout << "MASTER DIRECTORY:" << " " << argv[1] << endl;
        cout << "SLAVE DIRECTORY:"  << " " << argv[2] << endl;

        // MAKING ARGV MASTER E SLAVE

		argv_master[0] = "ossim-chipper";
		argv_master[1] = "--op";
		argv_master[2] = "ortho";
		argv_master[3] = argv[1];
		argv_master[4] = argv[3];

		argv_slave[0] =  "ossim-chipper";
		argv_slave[1] =  "--op";
		argv_slave[2] =  "ortho";
		argv_slave[3] = argv[2];
		argv_slave[4] = argv[4];

    int originalArgCount = 5;
		int originalArgCount2 = 5;

		if(argc == 10) 
		{
			argv_master[5] = argv[5];
			argv_master[6] = argv[6];
			argv_master[7] = argv[7];
			argv_master[8] = argv[8];
			argv_master[9] = argv[9];

			argv_slave[5] = argv[5];
			argv_slave[6] = argv[6];
			argv_slave[7] = argv[7];
			argv_slave[8] = argv[8];
			argv_slave[9] = argv[9];

			originalArgCount = 10;
			originalArgCount2 = 10;

        	cout << "TILE CUT:" << " " << "Lat_min" << " " << argv[6] 
        						<< " " << "Lon_min" << " " << argv[7]
        						<< " " << "Lat_max" << " " << argv[8]
        						<< " " << "Lon_max" << " " << argv[9] << endl;
		}	

		// ORTHORECTIFICATION

		cout << "Start master orthorectification" << endl;
		ossimArgumentParser ap_master(&originalArgCount, argv_master);
		ortho(ap_master); 
	
		cout << "Start slave orthorectification" << endl;
		ossimArgumentParser ap_slave(&originalArgCount2, argv_slave);
		ortho(ap_slave);
       
		// TP GENERATOR

        ossimImageHandler* master_handler = ossimImageHandlerRegistry::instance()->open(ossimFilename(argv[3]));
        ossimImageHandler* slave_handler = ossimImageHandlerRegistry::instance()->open(ossimFilename(argv[4]));
        if(master_handler && slave_handler) // enter if exist both master and slave  
      		{
      		ossimIrect bounds_master = master_handler->getBoundingRect(0);
      		ossimIrect bounds_slave = slave_handler->getBoundingRect(0);      		
      		ossimRefPtr<ossimImageData> img_master = master_handler->getTile(bounds_master, 0); 
      		ossimRefPtr<ossimImageData> img_slave = slave_handler->getTile(bounds_slave, 0); 
      		openCVtestclass* test = new openCVtestclass(img_master, img_slave);
      		test->run();
	   		  test->TPgen();
	   		  test->TPdraw();
	   		  test->warp();                  
			    }      
		      
          if(master_handler && slave_handler) // enter if exist both master and slave  
          {
          DisparityMap* map = new DisparityMap(img_master, img_slave);
          map->execute();
          } 

		  }
   catch (const ossimException& e)
		{
     	 	ossimNotify(ossimNotifyLevel_WARN) << e.what() << endl;
      		return 1;
		}
   
   return 0;
}

/*./bin/ossim-opencv ../../../../img_data/po_3800808_0000000/po_3800808_pan_0000000.tif ../../../../img_data/po_3800808_0010000/po_3800808_pan_0010000.tif ../../../../img_data/risultati_epipolar/ortho_ritaglio1.jpg ../../../../img_data/risultati_epipolar/ortho_ritaglio2.jpg --cut-bbox-ll 44.603 11.816 44.623 11.851 */