/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
const char _uuconf_thread_rcsid[] = "$Id: thread.c,v 1.6 2002/03/05 19:10:43 ian Rel $";
#endif

#include <errno.h>

/* Initialize for a new thread, by allocating a new sglobal structure
   which points to the same sprocess structure.  */

int
uuconf_init_thread (ppglobal)
     pointer *ppglobal;
{
  struct sglobal **pqglob = (struct sglobal **) ppglobal;
  pointer pblock;
  struct sglobal *qnew;

  pblock = uuconf_malloc_block ();
  if (pblock == NULL)
    {
      (*pqglob)->ierrno = errno;
      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
    }

  qnew = (struct sglobal *) uuconf_malloc (pblock,
					   sizeof (struct sglobal));
  if (qnew == NULL)
    {
      (*pqglob)->ierrno = errno;
      uuconf_free_block (pblock);
      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
    }

  qnew->pblock = pblock;
  qnew->ierrno = 0;
  qnew->ilineno = 0;
  qnew->zfilename = NULL;
  qnew->qprocess = (*pqglob)->qprocess;

  *pqglob = qnew;

  return UUCONF_SUCCESS;
}
