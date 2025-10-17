/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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

#include "cgi.h"


/*
 * 'main()' - Test the template code.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line arguments */
     char *argv[])			/* I - Command-line arguments */
{
  int	i;				/* Looping var */
  char	*value;				/* Value in name=value */
  FILE	*out;				/* Where to send output */


 /*
  * Don't buffer stdout or stderr so that the mixed output is sane...
  */

  setbuf(stdout, NULL);
  setbuf(stderr, NULL);

 /*
  * Loop through the command-line, assigning variables for any args with
  * "name=value"...
  */

  out = stdout;

  for (i = 1; i < argc; i ++)
  {
    if (!strcmp(argv[i], "-o"))
    {
      i ++;
      if (i < argc)
      {
        out = fopen(argv[i], "w");
	if (!out)
	{
	  perror(argv[i]);
	  return (1);
	}
      }
    }
    else if (!strcmp(argv[i], "-e"))
    {
      i ++;

      if (i < argc)
      {
        if (!freopen(argv[i], "w", stderr))
	{
	  perror(argv[i]);
	  return (1);
	}
      }
    }
    else if (!strcmp(argv[i], "-q"))
      freopen("/dev/null", "w", stderr);
    else if ((value = strchr(argv[i], '=')) != NULL)
    {
      *value++ = '\0';
      cgiSetVariable(argv[i], value);
    }
    else
      cgiCopyTemplateFile(out, argv[i]);
  }

 /*
  * Return with no errors...
  */

  return (0);
}
