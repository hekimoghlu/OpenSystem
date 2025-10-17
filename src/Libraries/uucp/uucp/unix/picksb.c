/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
const char picksb_rcsid[] = "$Id: picksb.c,v 1.13 2002/03/05 19:10:42 ian Rel $";
#endif

#include "uudefs.h"
#include "system.h"
#include "sysdep.h"

#include <errno.h>
#include <pwd.h>

#if HAVE_OPENDIR
#if HAVE_DIRENT_H
#include <dirent.h>
#else /* ! HAVE_DIRENT_H */
#include <sys/dir.h>
#define dirent direct
#endif /* ! HAVE_DIRENT_H */
#endif /* HAVE_OPENDIR */

#if GETPWUID_DECLARATION_OK
#ifndef getpwuid
extern struct passwd *getpwuid ();
#endif
#endif

/* Local variables.  */

/* Directory of ~/receive/USER.  */
static DIR *qStopdir;

/* Name of ~/receive/USER.  */
static char *zStopdir;

/* Directory of ~/receive/USER/SYSTEM.  */
static DIR *qSsysdir;

/* Name of system.  */
static char *zSsysdir;

/* Prepare to get a list of all the file to uupick for this user.  */

/*ARGSUSED*/
boolean
fsysdep_uupick_init (zsystem, zpubdir)
     const char *zsystem ATTRIBUTE_UNUSED;
     const char *zpubdir;
{
  const char *zuser;

  zuser = zsysdep_login_name ();

  zStopdir = (char *) xmalloc (strlen (zpubdir)
			       + sizeof "/receive/"
			       + strlen (zuser));
  sprintf (zStopdir, "%s/receive/%s", zpubdir, zuser);

  qStopdir = opendir (zStopdir);
  if (qStopdir == NULL && errno != ENOENT)
    {
      ulog (LOG_ERROR, "opendir (%s): %s", zStopdir,
	    strerror (errno));
      return FALSE;
    }

  qSsysdir = NULL;

  return TRUE;
}

/* Return the next file from the uupick directories.  */

/*ARGSUSED*/
char *
zsysdep_uupick (zsysarg, zpubdir, pzfrom, pzfull)
     const char *zsysarg;
     const char *zpubdir ATTRIBUTE_UNUSED;
     char **pzfrom;
     char **pzfull;
{
  struct dirent *qentry;

  while (TRUE)
    {
      while (qSsysdir == NULL)
	{
	  const char *zsystem;
	  char *zdir;

	  if (qStopdir == NULL)
	    return NULL;

	  if (zsysarg != NULL)
	    {
	      closedir (qStopdir);
	      qStopdir = NULL;
	      zsystem = zsysarg;
	    }
	  else
	    {
	      do
		{
		  qentry = readdir (qStopdir);
		  if (qentry == NULL)
		    {
		      closedir (qStopdir);
		      qStopdir = NULL;
		      return NULL;
		    }
		}
	      while (strcmp (qentry->d_name, ".") == 0
		     || strcmp (qentry->d_name, "..") == 0);

	      zsystem = qentry->d_name;
	    }

	  zdir = zbufalc (strlen (zStopdir) + strlen (zsystem) + sizeof "/");
	  sprintf (zdir, "%s/%s", zStopdir, zsystem);

	  qSsysdir = opendir (zdir);
	  if (qSsysdir == NULL)
	    {
	      if (errno != ENOENT && errno != ENOTDIR)
		ulog (LOG_ERROR, "opendir (%s): %s", zdir, strerror (errno));
	    }
	  else
	    {
	      ubuffree (zSsysdir);
	      zSsysdir = zbufcpy (zsystem);
	    }

	  ubuffree (zdir);
	}

      qentry = readdir (qSsysdir);
      if (qentry == NULL)
	{
	  closedir (qSsysdir);
	  qSsysdir = NULL;
	  continue;
	}

      if (strcmp (qentry->d_name, ".") == 0
	  || strcmp (qentry->d_name, "..") == 0)
	continue;

      *pzfrom = zbufcpy (zSsysdir);
      *pzfull = zsappend3 (zStopdir, zSsysdir, qentry->d_name);
      return zbufcpy (qentry->d_name);
    }
}

/*ARGSUSED*/
boolean
fsysdep_uupick_free (zsystem, zpubdir)
     const char *zsystem ATTRIBUTE_UNUSED;
     const char *zpubdir ATTRIBUTE_UNUSED;
{
  xfree ((pointer) zStopdir);
  if (qStopdir != NULL)
    {
      closedir (qStopdir);
      qStopdir = NULL;
    }
  ubuffree (zSsysdir);
  zSsysdir = NULL;
  if (qSsysdir != NULL)
    {
      closedir (qSsysdir);
      qSsysdir = NULL;
    }

  return TRUE;
}

/* Expand a local file name for uupick.  */

char *
zsysdep_uupick_local_file (zfile, pfbadname)
     const char *zfile;
     boolean *pfbadname;
{
  struct passwd *q;

  if (pfbadname != NULL)
    *pfbadname = FALSE;

  /* If this does not start with a simple ~, pass it to
     zsysdep_local_file_cwd; as it happens, zsysdep_local_file_cwd
     only uses the zpubdir argument if the file starts with a simple
     ~, so it doesn't really matter what we pass for zpubdir.  */
  if (zfile[0] != '~'
      || (zfile[1] != '/' && zfile[1] != '\0'))
    return zsysdep_local_file_cwd (zfile, (const char *) NULL, pfbadname);
  
  q = getpwuid (getuid ());
  if (q == NULL)
    {
      ulog (LOG_ERROR, "Can't get home directory");
      return NULL;
    }

  if (zfile[1] == '\0')
    return zbufcpy (q->pw_dir);

  return zsysdep_in_dir (q->pw_dir, zfile + 2);
}
