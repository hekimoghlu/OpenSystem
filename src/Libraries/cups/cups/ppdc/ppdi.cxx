/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//
// PPD file import utility for the CUPS PPD Compiler.
//
// Copyright 2007-2011 by Apple Inc.
// Copyright 2002-2005 by Easy Software Products.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more information.
//

//
// Include necessary headers...
//

#include "ppdc-private.h"
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>


//
// Local functions...
//

static void	usage(void);


//
// 'main()' - Main entry for the PPD import utility.
//

int					// O - Exit status
main(int  argc,				// I - Number of command-line arguments
     char *argv[])			// I - Command-line arguments
{
  int		i;			// Looping var
  char		*opt;			// Current option
  const char	*srcfile;		// Output file
  ppdcSource	*src;			// PPD source file data


  _cupsSetLocale(argv);

  // Scan the command-line...
  srcfile = NULL;
  src     = NULL;

  for (i = 1; i < argc; i ++)
    if (argv[i][0] == '-')
    {
      for (opt = argv[i] + 1; *opt; opt ++)
        switch (*opt)
	{
	  case 'o' :			// Output file
              if (srcfile || src)
	        usage();

	      i ++;
	      if (i >= argc)
        	usage();

	      srcfile = argv[i];
	      break;

	  case 'I' :			// Include dir
	      i ++;
	      if (i >= argc)
        	usage();

	      ppdcSource::add_include(argv[i]);
	      break;

	  default :			// Unknown
	      usage();
	      break;
        }
    }
    else
    {
      // Open and load the driver info file...
      if (!srcfile)
        srcfile = "ppdi.drv";

      if (!src)
      {
        if (access(srcfile, 0))
	  src = new ppdcSource();
	else
          src = new ppdcSource(srcfile);
      }

      // Import the PPD file...
      src->import_ppd(argv[i]);
    }

  // If no drivers have been loaded, display the program usage message.
  if (!src)
    usage();

  // Write the driver info file back to disk...
  src->write_file(srcfile);

  // Delete the printer driver information...
  src->release();

  // Return with no errors.
  return (0);
}


//
// 'usage()' - Show usage and exit.
//

static void
usage(void)
{
  _cupsLangPuts(stdout, _("Usage: ppdi [options] filename.ppd [ ... "
			  "filenameN.ppd ]"));
  _cupsLangPuts(stdout, _("Options:"));
  _cupsLangPuts(stdout, _("  -I include-dir          Add include directory to "
                          "search path."));
  _cupsLangPuts(stdout, _("  -o filename.drv         Set driver information "
                          "file (otherwise ppdi.drv)."));

  exit(1);
}
