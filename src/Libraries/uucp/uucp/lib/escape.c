/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#include "uucp.h"

#include <ctype.h>

#include "uudefs.h"

size_t
cescape (z)
     char *z;
{
  char *zto, *zfrom;
  
  zto = z;
  zfrom = z;
  while (*zfrom != '\0')
    {
      if (*zfrom != '\\')
	{
	  *zto++ = *zfrom++;
	  continue;
	}
      ++zfrom;
      switch (*zfrom)
	{
	case '-':
	  *zto++ = '-';
	  break;
	case 'b':
	  *zto++ = '\b';
	  break;
	case 'n':
	  *zto++ = '\n';
	  break;
	case 'N':
	  *zto++ = '\0';
	  break;
	case 'r':
	  *zto++ = '\r';
	  break;
	case 's':
	  *zto++ = ' ';
	  break;
	case 't':
	  *zto++ = '\t';
	  break;
	case '\0':
	  --zfrom;
	  /* Fall through.  */
	case '\\':
	  *zto++ = '\\';
	  break;
	case '0': case '1': case '2': case '3': case '4':
	case '5': case '6': case '7': case '8': case '9':
	  {
	    int i;

	    i = *zfrom - '0';
	    if (zfrom[1] >= '0' && zfrom[1] <= '7')
	      i = 8 * i + *++zfrom - '0';
	    if (zfrom[1] >= '0' && zfrom[1] <= '7')
	      i = 8 * i + *++zfrom - '0';
	    *zto++ = (char) i;
	  }
	  break;
	case 'x':
	  {
	    int i;

	    i = 0;
	    while (isxdigit (BUCHAR (zfrom[1])))
	      {
		if (isdigit (BUCHAR (zfrom[1])))
		  i = 16 * i + *++zfrom - '0';
		else if (isupper (BUCHAR (zfrom[1])))
		  i = 16 * i + *++zfrom - 'A' + 10;
		else
		  i = 16 * i + *++zfrom - 'a' + 10;
	      }
	    *zto++ = (char) i;
	  }
	  break;
	default:
	  ulog (LOG_ERROR, "Unrecognized escape sequence \\%c",
		*zfrom);
	  *zto++ = *zfrom;
	  break;
	}

      ++zfrom;
    }

  *zto = '\0';

  return (size_t) (zto - z);
}
