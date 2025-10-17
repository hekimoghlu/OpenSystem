/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
const char _uuconf_local_rcsid[] = "$Id: local.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

/* Get default information about the local system.  */

int
uuconf_system_local (pglobal, qsys)
     pointer pglobal;
     struct uuconf_system *qsys;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;

  _uuconf_uclear_system (qsys);
  qsys->uuconf_palloc = uuconf_malloc_block ();
  if (qsys->uuconf_palloc == NULL)
    {
      qglobal->ierrno = errno;
      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
    }

  qsys->uuconf_zname = (char *) qglobal->qprocess->zlocalname;

  /* By default, we permit the local system to forward to and from any
     system.  */
  iret = _uuconf_iadd_string (qglobal, (char *) "ANY", FALSE, FALSE,
			      &qsys->uuconf_pzforward_from,
			      qsys->uuconf_palloc);
  if (iret == UUCONF_SUCCESS)
    iret = _uuconf_iadd_string (qglobal, (char *) "ANY", FALSE, FALSE,
				&qsys->uuconf_pzforward_to,
				qsys->uuconf_palloc);
  if (iret != UUCONF_SUCCESS)
    {
      uuconf_free_block (qsys->uuconf_palloc);
      return iret;
    }

  return _uuconf_isystem_basic_default (qglobal, qsys);
}
