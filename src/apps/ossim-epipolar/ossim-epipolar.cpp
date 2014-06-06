//*******************************************************************
//
// License:  LGPL
// 
// See LICENSE.txt file in the top level directory for more details.
//
// Author:  Garrett Potts
//
//*******************************************************************
//  $Id: ossim-orthoigen.cpp 3023 2011-11-02 15:02:27Z david.burken $

#include <iostream>
#include <cstdlib>
#include <list>
#include <fstream>
#include <iterator>
#include <iomanip>
using namespace std;

#include <ossim/parallel/ossimOrthoIgen.h>
#include <ossim/parallel/ossimMpi.h>
#include <ossim/init/ossimInit.h>
#include <ossim/base/ossimException.h>
#include <ossim/base/ossimNotifyContext.h>
#include <ossim/base/ossimArgumentParser.h>
#include <ossim/base/ossimApplicationUsage.h>
#include <ossim/base/ossimTrace.h>
#include <ossim/base/ossimRefPtr.h>
#include <ossim/base/ossimTimer.h>
#include <ossim/imaging/ossimImageWriterFactoryRegistry.h>
#include <sstream>

#include <ossim/support_data/ossimSrtmSupportData.h>
#include <ossim/base/ossimKeywordlist.h>

static ossimTrace traceDebug("orthoigen:debug");

//*************************************************************************************************
// USAGE
//*************************************************************************************************
static void usage()
{
   ossimNotify(ossimNotifyLevel_NOTICE) <<
      "Valid output writer types for \"-w\" or \"--writer\" option:\n\n" << ends;
   ossimImageWriterFactoryRegistry::instance()->
      printImageTypeList(ossimNotify(ossimNotifyLevel_NOTICE));
}

//*************************************************************************************************
// FINALIZE -- Convenient location for placing debug breakpoint for catching program exit.
//*************************************************************************************************
void finalize(int code)
{
   exit (code);
}

//*************************************************************************************************
// MAIN
//*************************************************************************************************
int main(int argc, char* argv[])
{
   ossimInit::instance()->initialize(argc, argv);

   if (argc != 2)
   {
      cout << "usage:  " << argv[0] << " srtm_file" << endl;
      return 0;
   }

   ossimSrtmSupportData sd;
   if (sd.setFilename(ossimFilename(argv[1]), true))
   {
      cout << sd << endl;
      double t = 1000.13;

      cout << "prova get "<<sd.getMeanPixelValue()<< endl;
       
      ossimKeywordlist kwl;
      sd.getImageGeometry(kwl);
      cout << "geometry file:\n" << kwl << endl;
   }
   else
   {
      cout << "Could not open:  " << argv[1] << endl;
   }
   
   return 0;

}
