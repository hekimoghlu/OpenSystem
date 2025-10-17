/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include <ctype.h>
#include <errno.h>

#if HAVE_GLOB && ! HAVE_GLOB_H
#undef HAVE_GLOB
#define HAVE_GLOB 0
#endif

#if HAVE_GLOB
#include <glob.h>
#endif

/* Local variables to hold the wildcard in progress.  */

#if HAVE_GLOB
static glob_t sSglob;
static unsigned int iSglob;
#else
static char *zSwildcard_alloc;
static char *zSwildcard;
#endif

/* Start getting a wildcarded file spec.  Use the glob function if it
   is available, and otherwise use the shell.  */

boolean
fsysdep_wildcard_start (zfile)
     const char *zfile;
{
#if HAVE_GLOB

#if DEBUG > 0
  if (*zfile != '/')
    ulog (LOG_FATAL, "fsysdep_wildcard: %s: Can't happen", zfile);
#endif

  if (glob (zfile, 0, (int (*) ()) NULL, &sSglob) != 0)
    sSglob.gl_pathc = 0;
  iSglob = 0;
  return TRUE;

#else /* ! HAVE_GLOB */

  char *zcmd, *zto;
  const char *zfrom;
  size_t c;
  const char *azargs[4];
  FILE *e;
  pid_t ipid;

#if DEBUG > 0
  if (*zfile != '/')
    ulog (LOG_FATAL, "fsysdep_wildcard: %s: Can't happen", zfile);
#endif

  zSwildcard_alloc = NULL;
  zSwildcard = NULL;

  zcmd = zbufalc (sizeof ECHO_PROGRAM + sizeof " " + 2 * strlen (zfile));
  memcpy (zcmd, ECHO_PROGRAM, sizeof ECHO_PROGRAM - 1);
  zto = zcmd + sizeof ECHO_PROGRAM - 1;
  *zto++ = ' ';
  zfrom = zfile;
  while (*zfrom != '\0')
    {
      /* To avoid shell trickery, we quote all characters except
	 letters, digits, and wildcard specifiers.  We don't quote '/'
	 to avoid an Ultrix sh bug.  */
      if (! isalnum (*zfrom)
	  && *zfrom != '*'
	  && *zfrom != '?'
	  && *zfrom != '['
	  && *zfrom != ']'
	  && *zfrom != '/')
	*zto++ = '\\';
      *zto++ = *zfrom++;
    }
  *zto = '\0';

  azargs[0] = "/bin/sh";
  azargs[1] = "-c";
  azargs[2] = zcmd;
  azargs[3] = NULL;

  e = espopen (azargs, TRUE, &ipid);

  ubuffree (zcmd);

  if (e == NULL)
    {
      ulog (LOG_ERROR, "espopen: %s", strerror (errno));
      return FALSE;
    }

  zSwildcard_alloc = NULL;
  c = 0;
  if (getline (&zSwildcard_alloc, &c, e) <= 0)
    {
      xfree ((pointer) zSwildcard_alloc);
      zSwildcard_alloc = NULL;
    }

  if (ixswait ((unsigned long) ipid, ECHO_PROGRAM) != 0)
    {
      xfree ((pointer) zSwildcard_alloc);
      return FALSE;
    }

  if (zSwildcard_alloc == NULL)
    return FALSE;

  DEBUG_MESSAGE1 (DEBUG_EXECUTE,
		  "fsysdep_wildcard_start: got \"%s\"",
		  zSwildcard_alloc);

  zSwildcard = zSwildcard_alloc;

  return TRUE;

#endif /* ! HAVE_GLOB */
}

/* Get the next wildcard spec.  */

/*ARGSUSED*/
char *
zsysdep_wildcard (zfile)
     const char *zfile ATTRIBUTE_UNUSED;
{
#if HAVE_GLOB

  char *zret;

  if (iSglob >= sSglob.gl_pathc)
    return NULL;
  zret = zbufcpy (sSglob.gl_pathv[iSglob]);
  ++iSglob;
  return zret;
  
#else /* ! HAVE_GLOB */

  char *zret;

  if (zSwildcard_alloc == NULL || zSwildcard == NULL)
    return NULL;

  zret = zSwildcard;

  while (*zSwildcard != '\0' && ! isspace (BUCHAR (*zSwildcard)))
    ++zSwildcard;

  if (*zSwildcard != '\0')
    {
      *zSwildcard = '\0';
      ++zSwildcard;
      while (*zSwildcard != '\0' && isspace (BUCHAR (*zSwildcard)))
	++zSwildcard;
    }

  if (*zSwildcard == '\0')
    zSwildcard = NULL;

  return zbufcpy (zret);

#endif /* ! HAVE_GLOB */
}

/* Finish up getting wildcard specs.  */

boolean
fsysdep_wildcard_end ()
{
#if HAVE_GLOB
  globfree (&sSglob);
  return TRUE;
#else /* ! HAVE_GLOB */
  xfree ((pointer) zSwildcard_alloc);
  zSwildcard_alloc = NULL;
  zSwildcard = NULL;
  return TRUE;
#endif /* ! HAVE_GLOB */
}
