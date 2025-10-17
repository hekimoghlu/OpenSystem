/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#include "uudefs.h"

/* Define MALLOC_BUFFERS when compiling this file in order to more
   effectively use a debugging malloc library.  */

#ifndef MALLOC_BUFFERS

/* We keep a linked list of buffers.  The union is a hack because the
   default definition of offsetof, in uucp.h, takes the address of the
   field, and some C compilers will not let you take the address of an
   array.  */

struct sbuf
{
  struct sbuf *qnext;
  size_t c;
  union
    {
      char ab[4];
      char bdummy;
    }
  u;
};

static struct sbuf *qBlist;

/* Get a buffer of a given size.  The buffer is returned with the
   ubuffree function.  */

char *
zbufalc (c)
     size_t c;
{
  register struct sbuf *q;

  if (qBlist == NULL)
    {
      q = (struct sbuf *) xmalloc (sizeof (struct sbuf) + c - 4);
      q->c = c;
    }
  else
    {
      q = qBlist;
      qBlist = q->qnext;
      if (q->c < c)
	{
	  q = (struct sbuf *) xrealloc ((pointer) q,
					sizeof (struct sbuf) + c - 4);
	  q->c = c;
	}
    }
  return q->u.ab;
}

/* Free up a buffer back onto the linked list.  */

void
ubuffree (z)
     char *z;
{
  struct sbuf *q;
  /* The type of ioff should be size_t, but making it int avoids a bug
     in some versions of the HP/UX compiler, and will always work.  */
  int ioff;

  if (z == NULL)
    return;
  ioff = offsetof (struct sbuf, u);
  q = (struct sbuf *) (pointer) (z - ioff);

#ifdef DEBUG_BUFFER
  {
    struct sbuf *qlook;

    for (qlook = qBlist; qlook != NULL; qlook = qlook->qnext)
      {
	if (qlook == q)
	  {
	    ulog (LOG_ERROR, "ubuffree: Attempt to free buffer twice");
	    abort ();
	  }
      }
  }
#endif

  q->qnext = qBlist;
  qBlist = q;
}

#else /* MALLOC_BUFFERS */

char *
zbufalc (c)
     size_t c;
{
  return (char *) xmalloc (c);
}

/* Free up a buffer back onto the linked list.  */

void
ubuffree (z)
     char *z;
{
  free (z);
}

#endif /* MALLOC_BUFFERS */

/* Get a buffer holding a given string.  */

char *
zbufcpy (z)
     const char *z;
{
  size_t csize;
  char *zret;

  if (z == NULL)
    return NULL;
  csize = strlen (z) + 1;
  zret = zbufalc (csize);
  memcpy (zret, z, csize);
  return zret;
}
