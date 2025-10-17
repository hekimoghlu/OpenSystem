/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
const char _uuconf_tcalou_rcsid[] = "$Id: tcalou.c,v 1.9 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

static int icsys P((pointer pglobal, int argc, char **argv, pointer pvar,
		    pointer pinfo));

/* Find the callout login name and password for a system from the
   Taylor UUCP configuration files.  */

int
uuconf_taylor_callout (pglobal, qsys, pzlog, pzpass)
     pointer pglobal;
     const struct uuconf_system *qsys;
     char **pzlog;
     char **pzpass;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  boolean flookup;
  struct uuconf_cmdtab as[2];
  char **pz;
  int iret;
  pointer pinfo;

  *pzlog = NULL;
  *pzpass = NULL;

  flookup = FALSE;

  if (qsys->uuconf_zcall_login != NULL)
    {
      if (strcmp (qsys->uuconf_zcall_login, "*") == 0)
	flookup = TRUE;
      else
	{
	  *pzlog = strdup (qsys->uuconf_zcall_login);
	  if (*pzlog == NULL)
	    {
	      qglobal->ierrno = errno;
	      return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	    }
	}
    }

  if (qsys->uuconf_zcall_password != NULL)
    {
      if (strcmp (qsys->uuconf_zcall_password, "*") == 0)
	flookup = TRUE;
      else
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
    }

  if (! flookup)
    {
      if (*pzlog == NULL && *pzpass == NULL)
	return UUCONF_NOT_FOUND;
      return UUCONF_SUCCESS;
    }

  as[0].uuconf_zcmd = qsys->uuconf_zname;
  as[0].uuconf_itype = UUCONF_CMDTABTYPE_FN | 0;
  if (*pzlog == NULL)
    as[0].uuconf_pvar = (pointer) pzlog;
  else
    as[0].uuconf_pvar = NULL;
  as[0].uuconf_pifn = icsys;

  as[1].uuconf_zcmd = NULL;

  if (*pzpass == NULL)
    pinfo = (pointer) pzpass;
  else
    pinfo = NULL;

  iret = UUCONF_SUCCESS;

  for (pz = qglobal->qprocess->pzcallfiles; *pz != NULL; pz++)
    {
      FILE *e;

      e = fopen (*pz, "r");
      if (e == NULL)
	{
	  if (FNO_SUCH_FILE ())
	    continue;
	  qglobal->ierrno = errno;
	  iret = UUCONF_FOPEN_FAILED | UUCONF_ERROR_ERRNO;
	  break;
	}

      iret = uuconf_cmd_file (pglobal, e, as, pinfo,
			      (uuconf_cmdtabfn) NULL, 0,
			      qsys->uuconf_palloc);
      (void) fclose (e);

      if (iret != UUCONF_SUCCESS)
	break;
      if (*pzlog != NULL)
	break;
    }

  if (iret != UUCONF_SUCCESS)
    {
      qglobal->zfilename = *pz;
      return iret | UUCONF_ERROR_FILENAME;
    }

  if (*pzlog == NULL && *pzpass == NULL)
    return UUCONF_NOT_FOUND;

  return UUCONF_SUCCESS;
}

/* Copy the login name and password onto the heap and set the
   pointers.  The pzlog argument is passed in pvar, and the pzpass
   argument is passed in pinfo.  */

static int
icsys (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal;
     int argc;
     char **argv;
     pointer pvar;
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  char **pzlog = (char **) pvar;
  char **pzpass = (char **) pinfo;

  if (argc < 2 || argc > 3)
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  if (pzlog != NULL)
    {
      *pzlog = strdup (argv[1]);
      if (*pzlog == NULL)
	{
	  qglobal->ierrno = errno;
	  return (UUCONF_MALLOC_FAILED
		  | UUCONF_ERROR_ERRNO
		  | UUCONF_CMDTABRET_EXIT);
	}
    }

  if (pzpass != NULL)
    {
      if (argc < 3)
	*pzpass = strdup ("");
      else
	*pzpass = strdup (argv[2]);
      if (*pzpass == NULL)
	{
	  qglobal->ierrno = errno;
	  if (pzlog != NULL)
	    {
	      free ((pointer) *pzlog);
	      *pzlog = NULL;
	    }
	  return (UUCONF_MALLOC_FAILED
		  | UUCONF_ERROR_ERRNO
		  | UUCONF_CMDTABRET_EXIT);
	}
    }

  return UUCONF_CMDTABRET_EXIT;
}
