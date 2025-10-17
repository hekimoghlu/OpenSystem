/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
const char _uuconf_addblk_rcsid[] = "$Id: addblk.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

#include "alloc.h"

/* Add a memory buffer allocated by malloc to a memory block.  This is
   used by the uuconf_cmd functions so that they don't have to
   constantly copy data into memory.  Returns 0 on success, non 0 on
   failure. */

int
uuconf_add_block (pblock, padd)
     pointer pblock;
     pointer padd;
{
  struct sblock *q = (struct sblock *) pblock;
  struct sadded *qnew;

  qnew = (struct sadded *) uuconf_malloc (pblock, sizeof (struct sadded));
  if (qnew == NULL)
    return 1;

  qnew->qnext = q->qadded;
  qnew->padded = padd;
  q->qadded = qnew;

  return 0;
}
