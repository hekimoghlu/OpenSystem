/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
const char _uuconf_free_rcsid[] = "$Id: free.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include "alloc.h"

/* Free memory allocated by uuconf_malloc.  If the memory block is
   NULL, this just calls free; this is convenient for a number of
   routines.  Otherwise, this will only do something if this was the
   last buffer allocated for one of the memory blocks in the list; in
   other cases, the memory is lost until the entire memory block is
   freed.  */

#if UUCONF_ANSI_C
void
#endif
uuconf_free (pblock, pbuf)
     pointer pblock;
     pointer pbuf;
{
  struct sblock *q = (struct sblock *) pblock;

  if (pbuf == NULL)
    return;

  if (q == NULL)
    {
      free (pbuf);
      return;
    }

  for (; q != NULL; q = q->qnext)
    {
      if (q->plast == pbuf)
	{
	  q->ifree = (char *) pbuf - q->u.ab;
	  /* We could reset q->plast here, but it doesn't matter.  */
	  return;
	}
    }
}
