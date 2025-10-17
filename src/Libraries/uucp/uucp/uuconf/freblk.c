/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
const char _uuconf_freblk_rcsid[] = "$Id: freblk.c,v 1.8 2002/03/05 19:10:42 ian Rel $";
#endif

#include "alloc.h"

/* Free up an entire memory block.  */

#if UUCONF_ANSI_C
void
#endif
uuconf_free_block (pblock)
     pointer pblock;
{
  struct sblock *q = (struct sblock *) pblock;
  struct sblock *qloop;

  /* We have to free the added blocks first because the list may link
     into blocks that are earlier on the list.  */
  for (qloop = q; qloop != NULL; qloop = qloop->qnext)
    {
      struct sadded *qadd;

      for (qadd = qloop->qadded; qadd != NULL; qadd = qadd->qnext)
	free (qadd->padded);
    }

  while (q != NULL)
    {
      struct sblock *qnext;

      qnext = q->qnext;
      free ((pointer) q);
      q = qnext;
    }
}
