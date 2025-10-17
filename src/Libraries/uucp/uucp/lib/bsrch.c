/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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

/* Perform a binary search for KEY in BASE which has NMEMB elements
   of SIZE bytes each.  The comparisons are done by (*COMPAR)().  */
pointer
bsearch (key, base, nmemb, size, compar)
     register constpointer key;
     register constpointer base;
     size_t nmemb;
     register size_t size;
     register int (*compar) P((constpointer, constpointer));
{
  register size_t l, u, idx;
  register constpointer p;
  register int comparison;

  l = 0;
  u = nmemb;
  while (l < u)
    {
      idx = (l + u) / 2;
      p = (constpointer) (((const char *) base) + (idx * size));
      comparison = (*compar)(key, p);
      if (comparison < 0)
	u = idx;
      else if (comparison > 0)
	l = idx + 1;
      else
	return (pointer) p;
    }

  return NULL;
}
