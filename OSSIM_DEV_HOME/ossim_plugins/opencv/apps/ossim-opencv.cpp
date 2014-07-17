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

bool ortho (ossimArgumentParser argPars )
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

int main(int argc, char *argv[])
{
   // Initialize ossim stuff, factories, plugin, etc.
   ossimTimer::instance()->setStartTick();
  /*	ossimArgumentParser ap(&argc, argv);
   		ossimInit::instance()->addOptions(ap);
   		ossimInit::instance()->initialize(ap);
-----*/
   	try
   		{  
      	// Put your code here.
        int originalArgCount = 5;
    	char * pluto[5];
		//pluto è una stringa di array di array di caratteri
		pluto[0] = "ossim-chipper";
		pluto[1] = "--op";
		pluto[2] = "ortho";
		//pluto[3] = "-P ";
		//pluto[4] = "../../../../preferences/ossim_prefs.txt";
		pluto[3] = "../../../../../img_data/po_3800808_0000000/po_3800808_pan_0000000.tif";
		pluto[4] = "../../../../../img_data/risultati_epipolar/ortho_ouput1.jpg";

 		cout << "N arg = " << argc << endl;
		cout << argv[0] << endl;
		cout << pluto[3] << endl;
		cout << pluto[4] << endl;

		ossimArgumentParser ap_master(&originalArgCount, pluto);
		cout << "Start master orthorectification" << endl;
		//ortho(ap_master); //sto chiamando la funzione, dandogli ap come input, che poi lui sostituirà con argPars dato che gli ho detto che ortho prende argPars (è come se ci fosse argPars=ap)
		// Initialize ossim stuff, factories, plugin, etc.
		//ossimInit::instance()->initialize(ap_master);

		//gli do nuova img in input e gli dico nuovo output
		int originalArgCount2 = 5;
		char * pluto2[5];
		pluto2[0] = "ossim-chipper";
		pluto2[1] = "--op";
		pluto2[2] = "ortho";
		//pluto2[3] = "-P";
		//pluto2[4] = "../../../../preferences/ossim_prefs.txt";
		pluto2[3] = "../../../../../img_data/po_3800808_0010000/po_3800808_pan_0010000.tif";
		pluto2[4] = "../../../../../img_data/risultati_epipolar/ortho_ouput2.jpg";
   
		//definisco ap_slave
		ossimArgumentParser ap_slave(&originalArgCount2, pluto2);
		cout << "Start slave orthorectification" << endl;
		//ortho(ap_slave);

		// Initialize ossim stuff, factories, plugin, etc.
		//ossimInit::instance()->initialize(ap_slave);
       
        ossimImageHandler* master_handler = ossimImageHandlerRegistry::instance()->open(ossimFilename("../../../../../img_data/risultati_epipolar/ortho_ouput1.jpg"));
        ossimImageHandler* slave_handler = ossimImageHandlerRegistry::instance()->open(ossimFilename("../../../../../img_data/risultati_epipolar/ortho_ouput2.jpg"));
        if(master_handler && slave_handler) //se esistono sia master che slave
      		{
      		ossimIrect bounds_master = master_handler->getBoundingRect(0);
      		ossimIrect bounds_slave = slave_handler->getBoundingRect(0);      		
      		ossimRefPtr<ossimImageData> img_master = master_handler->getTile(bounds_master, 0); 
      		ossimRefPtr<ossimImageData> img_slave = slave_handler->getTile(bounds_slave, 0); 
      		openCVtestclass* test = new openCVtestclass(img_master, img_slave);
      		test->run();

	   		test->TPgen();
	   		test->TPdraw();
			}      
    	
		// test è puntatore ad oggetto di tipo openCVtestclass, mi permette di accedere ai metodi ed alle variabili della classe 
       	// con new sto istanziando la classe, richiamando il costruttore	  
		
		}
   catch (const ossimException& e)
		{
     	 	ossimNotify(ossimNotifyLevel_WARN) << e.what() << endl;
      		return 1;
		}
   
   return 0;
}
