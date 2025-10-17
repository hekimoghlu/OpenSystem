/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
/*
 * Include necessary headers...
 */

#include <stdio.h>
#include <string.h>
#include "ppdx.h"


/*
 * 'main()' - Read data from a test PPD file and write out new chunks.
 */

int					/* O - Exit status */
main(void)
{
  int		status = 0;		/* Exit status */
  FILE		*fp;			/* File to read */
  char		contents[8193],		/* Contents of file */
		*data;			/* Data from PPD */
  size_t	contsize,		/* File size */
		datasize;		/* Data size */
  ppd_file_t	*ppd;			/* Test PPD */


 /*
  * Open the PPD and get the data from it...
  */

  ppd  = ppdOpenFile("testppdx.ppd");
  data = ppdxReadData(ppd, "EXData", &datasize);

 /*
  * Open this source file and read it...
  */

  fp = fopen("testppdx.c", "r");
  if (fp)
  {
    contsize = fread(contents, 1, sizeof(contents) - 1, fp);
    fclose(fp);
    contents[contsize] = '\0';
  }
  else
  {
    contents[0] = '\0';
    contsize    = 0;
  }

 /*
  * Compare data...
  */

  if (data)
  {
    if (contsize != datasize)
    {
      fprintf(stderr, "ERROR: PPD has %ld bytes, test file is %ld bytes.\n",
              (long)datasize, (long)contsize);
      status = 1;
    }
    else if (strcmp(contents, data))
    {
      fputs("ERROR: PPD and test file are not the same.\n", stderr);
      status = 1;
    }

    if (status)
    {
      if ((fp = fopen("testppdx.dat", "wb")) != NULL)
      {
        fwrite(data, 1, datasize, fp);
        fclose(fp);
        fputs("ERROR: See testppdx.dat for data from PPD.\n", stderr);
      }
      else
        perror("Unable to open 'testppdx.dat'");
    }

    free(data);
  }

  printf("Encoding %ld bytes for PPD...\n", (long)contsize);

  ppdxWriteData("EXData", contents, contsize);

  return (1);
}
