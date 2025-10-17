/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
const char _uuconf_calout_rcsid[] = "$Id: calout.c,v 1.7 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

/* Find callout login name and password for a system.  */

/*ARGSUSED*/
int
uuconf_callout (pglobal, qsys, pzlog, pzpass)
     pointer pglobal;
     const struct uuconf_system *qsys;
     char **pzlog;
     char **pzpass;
{
#if HAVE_TAYLOR_CONFIG

  return uuconf_taylor_callout (pglobal, qsys, pzlog, pzpass);

#else /* ! HAVE_TAYLOR_CONFIG */

  struct sglobal *qglobal = (struct sglobal *) pglobal;

  *pzlog = NULL;
  *pzpass = NULL;

  if (qsys->uuconf_zcall_login == NULL
      && qsys->uuconf_zcall_password == NULL)
    return UUCONF_NOT_FOUND;

  if ((qsys->uuconf_zcall_login != NULL
       && strcmp (qsys->uuconf_zcall_login, "*") == 0)
      || (qsys->uuconf_zcall_password != NULL
	  && strcmp (qsys->uuconf_zcall_password, "*") == 0))
    return UUCONF_NOT_FOUND;
      
  if (qsys->uuconf_zcall_login != NULL)
    {
      *pzlog = strdup (qsys->uuconf_zcall_login);
      if (*pzlog == NULL)
	{
	  qglobal->ierrno = errno;
	  return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	}
    }

  if (qsys->uuconf_zcall_password != NULL)
    {
      *pzpass = strdup (qsys->uuconf_zcall_password);
      if (*pzpass == NULL)
	{
	  qglobal->ierrno = errno;
	  if (*pzlog != NULL)
	    {
	      free ((pointer) *pzlog);
	      *pzlog = NULL;
	    }
	  return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	}
    }

  return UUCONF_SUCCESS;

#endif /* ! HAVE_TAYLOR_CONFIG */
}
