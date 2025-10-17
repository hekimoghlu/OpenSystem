/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
const char _uuconf_tgcmp_rcsid[] = "$Id: tgcmp.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

/* A comparison function to pass to _uuconf_itime_parse.  This
   compares grades.  We can't just pass uuconf_grade_cmp, since
   _uuconf_itime_parse wants a function takes longs as arguments.  */

int
_uuconf_itime_grade_cmp (i1, i2)
     long i1;
     long i2;
{
  return UUCONF_GRADE_CMP ((int) i1, (int) i2);
}
