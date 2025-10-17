/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#include <cups/cups-private.h>


/*
 * 'main()' - Copy (and uncompress) files to stdout.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line arguments */
     char *argv[])			/* I - Command-line arguments */
{
  cups_file_t	*fp;			/* File */
  char		buffer[8192];		/* Data buffer */
  ssize_t	bytes;			/* Number of bytes read/written */
  int		copies;			/* Number of copies */


 /*
  * Check command-line...
  */

  if (argc < 6 || argc > 7)
  {
    _cupsLangPrintf(stderr,
                    _("Usage: %s job-id user title copies options [file]"),
                    argv[0]);
    return (1);
  }

 /*
  * Get the copy count; if we have no final content type, this is a
  * raw queue or raw print file, so we need to make copies...
  */

  if (!getenv("FINAL_CONTENT_TYPE"))
    copies = atoi(argv[4]);
  else
    copies = 1;

 /*
  * Open the file...
  */

  if (argc == 6)
  {
    copies = 1;
    fp     = cupsFileStdin();
  }
  else if ((fp = cupsFileOpen(argv[6], "r")) == NULL)
  {
    fprintf(stderr, "DEBUG: Unable to open \"%s\".\n", argv[6]);
    _cupsLangPrintError("ERROR", _("Unable to open print file"));
    return (1);
  }

 /*
  * Copy the file to stdout...
  */

  while (copies > 0)
  {
    if (!getenv("FINAL_CONTENT_TYPE"))
      fputs("PAGE: 1 1\n", stderr);

    cupsFileRewind(fp);

    while ((bytes = cupsFileRead(fp, buffer, sizeof(buffer))) > 0)
      if (write(1, buffer, (size_t)bytes) < bytes)
      {
	_cupsLangPrintFilter(stderr, "ERROR",
			     _("Unable to write uncompressed print data: %s"),
			     strerror(errno));
        if (argc == 7)
	  cupsFileClose(fp);

	return (1);
      }

    copies --;
  }

 /*
  * Close the file and return...
  */

  if (argc == 7)
    cupsFileClose(fp);

  return (0);
}
