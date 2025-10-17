/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include "sysdep.h"
#include "system.h"

#include <errno.h>

/* See whether a file is in a directory, and optionally check access.  */

boolean
fsysdep_in_directory (zfile, zdir, fcheck, freadable, zuser)
     const char *zfile;
     const char *zdir;
     boolean fcheck;
     boolean freadable;
     const char *zuser;
{
  size_t c;
  char *zcopy, *zslash;
  struct stat s;

  if (*zfile != '/')
    return FALSE;
  c = strlen (zdir);
  if (c > 0 && zdir[c - 1] == '/')
    c--;
  if (strncmp (zfile, zdir, c) != 0
      || (zfile[c] != '/' && zfile[c] != '\0'))
    return FALSE;
  if (strstr (zfile + c, "/../") != NULL)
    return FALSE;

  /* If we're not checking access, get out now.  */
  if (! fcheck)
    return TRUE;

  zcopy = zbufcpy (zfile);

  /* Start checking directories after zdir.  Otherwise, we would
     require that all directories down to /usr/spool/uucppublic be
     publically searchable; they probably are but it should not be a
     requirement.  */
  zslash = zcopy + c;
  do
    {
      char b;
      struct stat shold;

      b = *zslash;
      *zslash = '\0';

      shold = s;
      if (stat (zcopy, &s) != 0)
	{
	  if (errno != ENOENT)
	    {
	      ulog (LOG_ERROR, "stat (%s): %s", zcopy, strerror (errno));
	      ubuffree (zcopy);
	      return FALSE;
	    }

	  /* If this is the top directory, any problems will be caught
	     later when we try to open it.  */
	  if (zslash == zcopy + c)
	    {
	      ubuffree (zcopy);
	      return TRUE;
	    }

	  /* Go back and check the last directory for read or write
	     access.  */
	  s = shold;
	  break;
	}

      /* If this is not a directory, get out of the loop.  */
      if (! S_ISDIR (s.st_mode))
	break;

      /* Make sure the directory is searchable.  */
      if (! fsuser_access (&s, X_OK, zuser))
	{
	  ulog (LOG_ERROR, "%s: %s", zcopy, strerror (EACCES));
	  ubuffree (zcopy);
	  return FALSE;
	}

      /* If we've reached the end of the string, get out.  */
      if (b == '\0')
	break;

      *zslash = b;
    }
  while ((zslash = strchr (zslash + 1, '/')) != NULL);

  /* At this point s holds a stat on the last component of the path.
     We must check it for readability or writeability.  */
  if (! fsuser_access (&s, freadable ? R_OK : W_OK, zuser))
    {
      ulog (LOG_ERROR, "%s: %s", zcopy, strerror (EACCES));
      ubuffree (zcopy);
      return FALSE;
    }

  ubuffree (zcopy);
  return TRUE;
}
