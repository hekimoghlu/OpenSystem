/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
const char _uuconf_mrgblk_rcsid[] = "$Id: mrgblk.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

#include "alloc.h"

/* Merge one memory block into another one, returning the combined
   memory block.  */

pointer
_uuconf_pmalloc_block_merge (p1, p2)
     pointer p1;
     pointer p2;
{
  struct sblock *q1 = (struct sblock *) p1;
  struct sblock *q2 = (struct sblock *) p2;
  struct sblock **pq;

  for (pq = &q1; *pq != NULL; pq = &(*pq)->qnext)
    ;
  *pq = q2;
  return (pointer) q1;
}
