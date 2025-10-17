/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

int
strncasecmp (z1, z2, c)
     const char *z1;
     const char *z2;
     size_t c;
{
  char b1, b2;

  if (c == 0)
    return 0;
  while ((b1 = *z1++) != '\0')
    {
      b2 = *z2++;
      if (b2 == '\0')
	return 1;
      if (b1 != b2)
	{
	  if (isupper (BUCHAR (b1)))
	    b1 = tolower (BUCHAR (b1));
	  if (isupper (BUCHAR (b2)))
	    b2 = tolower (BUCHAR (b2));
	  if (b1 != b2)
	    return b1 - b2;
	}
      --c;
      if (c == 0)
	return 0;
    }
  if (*z2 == '\0')
    return 0;
  else
    return -1;
}
