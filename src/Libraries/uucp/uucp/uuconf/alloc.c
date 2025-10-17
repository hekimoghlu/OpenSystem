/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
const char _uuconf_alloc_rcsid[] = "$Id: alloc.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

#include "alloc.h"

/* Allocate some memory out of a memory block.  If the memory block is
   NULL, this just calls malloc; this is convenient for a number of
   routines.  If this fails, uuconf_errno will be set, and the calling
   routine may return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO.  */

pointer
uuconf_malloc (pblock, c)
     pointer pblock;
     size_t c;
{
  struct sblock *q = (struct sblock *) pblock;
  pointer pret;

  if (c == 0)
    return NULL;

  if (q == NULL)
    return malloc (c);

  /* Make sure that c is aligned to a double boundary.  */
  c = ((c + sizeof (double) - 1) / sizeof (double)) * sizeof (double);

  while (q->ifree + c > CALLOC_SIZE)
    {
      if (q->qnext != NULL)
	q = q->qnext;
      else
	{
	  if (c > CALLOC_SIZE)
	    q->qnext = (struct sblock *) malloc (sizeof (struct sblock)
						 + c - CALLOC_SIZE);
	  else
	    q->qnext = (struct sblock *) malloc (sizeof (struct sblock));
	  if (q->qnext == NULL)
	    return NULL;
	  q = q->qnext;
	  q->qnext = NULL;
	  q->ifree = 0;
	  q->qadded = NULL;
	  break;
	}
    }

  pret = q->u.ab + q->ifree;
  q->ifree += c;
  q->plast = pret;

  return pret;
}
