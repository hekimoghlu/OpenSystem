/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#include "uucnfi.h"

#if USE_RCS_ID
const char _uuconf_grdcmp_rcsid[] = "$Id: grdcmp.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

#include <ctype.h>

/* Compare two grades, returning < 0 if b1 should be executed before
   b2, == 0 if they are the same, or > 0 if b1 should be executed
   after b2.  This can not fail, and does not return a standard uuconf
   error code.

   This implementation assumes that the upper case letters are in
   sequence, and that the lower case letters are in sequence.  */

int
uuconf_grade_cmp (barg1, barg2)
     int barg1;
     int barg2;
{
  int b1, b2;

  /* Make sure the arguments are unsigned.  */
  b1 = (int) BUCHAR (barg1);
  b2 = (int) BUCHAR (barg2);

  if (isdigit (b1))
    {
      if (isdigit (b2))
	return b1 - b2;
      else
	return -1;
    }
  else if (isupper (b1))
    {
      if (isdigit (b2))
	return 1;
      else if (isupper (b2))
	return b1 - b2;
      else
	return -1;
    }
  else
    {
      if (! islower (b2))
	return 1;
      else
	return b1 - b2;
    }
}
