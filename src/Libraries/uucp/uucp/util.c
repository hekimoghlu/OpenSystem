/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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

#if USE_RCS_ID
const char util_rcsid[] = "$Id: util.c,v 1.11 2002/03/05 19:10:42 ian Rel $";
#endif

#include <ctype.h>

#include "uudefs.h"
#include "uuconf.h"
#include "system.h"

/* Get information for an unknown system.  This will leave the name
   allocated on the heap.  We could fix this by breaking the
   abstraction and adding the name to qsys->palloc.  It makes sure the
   name is not too long, but takes no other useful action.  */

boolean
funknown_system (puuconf, zsystem, qsys)
     pointer puuconf;
     const char *zsystem;
     struct uuconf_system *qsys;
{
  char *z;
  int iuuconf;

  if (strlen (zsystem) <= cSysdep_max_name_len)
    z = zbufcpy (zsystem);
  else
    {
      char **pznames, **pz;
      boolean ffound;

      z = zbufalc (cSysdep_max_name_len + 1);
      memcpy (z, zsystem, cSysdep_max_name_len);
      z[cSysdep_max_name_len] = '\0';

      iuuconf = uuconf_system_names (puuconf, &pznames, TRUE);
      if (iuuconf != UUCONF_SUCCESS)
	ulog_uuconf (LOG_FATAL, puuconf, iuuconf);

      ffound = FALSE;
      for (pz = pznames; *pz != NULL; pz++)
	{
	  if (strcmp (*pz, z) == 0)
	    ffound = TRUE;
	  xfree ((pointer) *pz);
	}
      xfree ((pointer) pznames);

      if (ffound)
	{
	  ubuffree (z);
	  return FALSE;
	}
    }

  iuuconf = uuconf_system_unknown (puuconf, qsys);
  if (iuuconf == UUCONF_NOT_FOUND)
    {
      ubuffree (z);
      return FALSE;
    }
  else if (iuuconf != UUCONF_SUCCESS)
    ulog_uuconf (LOG_FATAL, puuconf, iuuconf);

  for (; qsys != NULL; qsys = qsys->uuconf_qalternate)
    qsys->uuconf_zname = z;

  return TRUE;
}

/* Remove all occurrences of the local system name followed by an
   exclamation point from the front of a string, returning the new
   string.  This is used by uucp and uux.  */

char *
zremove_local_sys (qlocalsys, z)
     struct uuconf_system *qlocalsys;
     char *z;
{
  size_t clen;
  char *zexclam;

  clen = strlen (qlocalsys->uuconf_zname);
  zexclam = strchr (z, '!');
  while (zexclam != NULL)
    {
      if (z == zexclam
	  || ((size_t) (zexclam - z) == clen
	      && strncmp (z, qlocalsys->uuconf_zname, clen) == 0))
	;
      else if (qlocalsys->uuconf_pzalias == NULL)
	break;
      else
	{
	  char **pzal;

	  for (pzal = qlocalsys->uuconf_pzalias; *pzal != NULL; pzal++)
	    if (strlen (*pzal) == (size_t) (zexclam - z)
		&& strncmp (z, *pzal, (size_t) (zexclam - z)) == 0)
	      break;
	  if (*pzal == NULL)
	    break;
	}
      z = zexclam + 1;
      zexclam = strchr (z, '!');
    }

  return z;
}

/* See whether a file is in a directory list, and make sure the user
   has appropriate access.  */

boolean
fin_directory_list (zfile, pzdirs, zpubdir, fcheck, freadable, zuser)
     const char *zfile;
     char **pzdirs;
     const char *zpubdir;
     boolean fcheck;
     boolean freadable;
     const char *zuser;
{
  boolean fmatch;
  char **pz;

  fmatch = FALSE;

  for (pz = pzdirs; *pz != NULL; pz++)
    {
      char *zuse;

      if (pz[0][0] == '!')
	{
	  zuse = zsysdep_local_file (*pz + 1, zpubdir, (boolean *) NULL);
	  if (zuse == NULL)
	    return FALSE;

	  if (fsysdep_in_directory (zfile, zuse, FALSE,
				    FALSE, (const char *) NULL))
	    fmatch = FALSE;
	}
      else
	{
	  zuse = zsysdep_local_file (*pz, zpubdir, (boolean *) NULL);
	  if (zuse == NULL)
	    return FALSE;

	  if (fsysdep_in_directory (zfile, zuse, fcheck,
				    freadable, zuser))
	    fmatch = TRUE;
	}

      ubuffree (zuse);
    }

  return fmatch;
}
