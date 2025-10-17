/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
const char xqtfil_rcsid[] = "$Id: xqtfil.c,v 1.13 2002/03/05 19:10:42 ian Rel $";
#endif

#include "uudefs.h"
#include "sysdep.h"
#include "system.h"

#include <errno.h>

#if HAVE_OPENDIR
#if HAVE_DIRENT_H
#include <dirent.h>
#else /* ! HAVE_DIRENT_H */
#include <sys/dir.h>
#define dirent direct
#endif /* ! HAVE_DIRENT_H */
#endif /* HAVE_OPENDIR */

/* Under the V2 or BSD42 spool directory scheme, all execute files are
   in the main spool directory.  Under the BSD43 scheme, they are all
   in the directory X..  Under the HDB or SVR4 scheme, they are in
   directories named after systems.  Under the ULTRIX scheme, they are
   in X.  subdirectories of subdirectories of sys.  Under the TAYLOR
   scheme, they are all in the subdirectory X. of a directory named
   after the system.

   This means that for HDB, ULTRIX, SVR4 or TAYLOR, we have to search
   directories of directories.  */

#if SPOOLDIR_V2 || SPOOLDIR_BSD42
#define ZDIR "."
#define SUBDIRS 0
#endif
#if SPOOLDIR_HDB || SPOOLDIR_SVR4 || SPOOLDIR_TAYLOR
#define ZDIR "."
#define SUBDIRS 1
#endif
#if SPOOLDIR_ULTRIX
#define ZDIR "sys"
#define SUBDIRS 1
#endif
#if SPOOLDIR_BSD43
#define ZDIR "X."
#define SUBDIRS 0
#endif

/* Static variables for the execute file scan.  */

static DIR *qSxqt_topdir;
#if ! SUBDIRS
static const char *zSdir;
#else /* SUBDIRS */
static boolean fSone_dir;
static char *zSdir;
static DIR *qSxqt_dir;
static char *zSsystem;
#endif /* SUBDIRS */

/* Initialize the scan for execute files.  The function
   usysdep_get_xqt_free will clear the data out when we are done with
   the system.  This returns FALSE on error.  */

/*ARGSUSED*/
boolean
fsysdep_get_xqt_init (zsystem)
     const char *zsystem;
{
  usysdep_get_xqt_free ((const char *) NULL);

#if SUBDIRS
  if (zsystem != NULL)
    {
#if SPOOLDIR_HDB || SPOOLDIR_SVR4
      zSdir = zbufcpy (zsystem);
#endif
#if SPOOLDIR_ULTRIX
      zSdir = zsappend3 ("sys", zsystem, "X.");
#endif
#if SPOOLDIR_TAYLOR
      zSdir = zsysdep_in_dir (zsystem, "X.");
#endif

      qSxqt_dir = opendir ((char *) zSdir);
      if (qSxqt_dir != NULL)
	{
	  qSxqt_topdir = qSxqt_dir;
	  fSone_dir = TRUE;
	  zSsystem = zbufcpy (zsystem);
	  return TRUE;
	}
    }

  fSone_dir = FALSE;
#endif

  qSxqt_topdir = opendir ((char *) ZDIR);
  if (qSxqt_topdir == NULL)
    {
      if (errno == ENOENT)
	return TRUE;
      ulog (LOG_ERROR, "opendir (%s): %s", ZDIR, strerror (errno));
      return FALSE;
    }

  return TRUE;
}

/* Return the name of the next execute file to read and process.  If
   this returns NULL, *pferr must be checked.  If will be TRUE on
   error, FALSE if there are no more files.  On a successful return
   *pzsystem will be set to the system for which the execute file was
   created.  */

/*ARGSUSED*/
char *
zsysdep_get_xqt (zsystem, pzsystem, pferr)
     const char *zsystem ATTRIBUTE_UNUSED;
     char **pzsystem;
     boolean *pferr;
{
  *pferr = FALSE;

  if (qSxqt_topdir == NULL)
    return NULL;

  /* This loop continues until we find a file.  */
  while (TRUE)
    {
      DIR *qdir;
      struct dirent *q;

#if ! SUBDIRS
      zSdir = ZDIR;
      qdir = qSxqt_topdir;
#else /* SUBDIRS */
      /* This loop continues until we find a subdirectory to read.  */
      while (qSxqt_dir == NULL)
	{
	  struct dirent *qtop;

	  qtop = readdir (qSxqt_topdir);
	  if (qtop == NULL)
	    {
	      (void) closedir (qSxqt_topdir);
	      qSxqt_topdir = NULL;
	      return NULL;
	    }

	  /* No system name may start with a dot This allows us to
	     quickly skip impossible directories.  */
	  if (qtop->d_name[0] == '.')
	    continue;

	  DEBUG_MESSAGE1 (DEBUG_SPOOLDIR,
			  "zsysdep_get_xqt: Found %s in top directory",
			  qtop->d_name);

	  ubuffree (zSdir);

#if SPOOLDIR_HDB || SPOOLDIR_SVR4
	  zSdir = zbufcpy (qtop->d_name);
#endif
#if SPOOLDIR_ULTRIX
	  zSdir = zsappend3 ("sys", qtop->d_name, "X.");
#endif
#if SPOOLDIR_TAYLOR
	  zSdir = zsysdep_in_dir (qtop->d_name, "X.");
#endif

	  ubuffree (zSsystem);
	  zSsystem = zbufcpy (qtop->d_name);

	  qSxqt_dir = opendir (zSdir);

	  if (qSxqt_dir == NULL
	      && errno != ENOTDIR
	      && errno != ENOENT)
	    ulog (LOG_ERROR, "opendir (%s): %s", zSdir, strerror (errno));
	}

      qdir = qSxqt_dir;
#endif /* SUBDIRS */

      q = readdir (qdir);

#if DEBUG > 1
      if (q != NULL)
	DEBUG_MESSAGE2 (DEBUG_SPOOLDIR,
			"zsysdep_get_xqt: Found %s in subdirectory %s",
			q->d_name, zSdir);
#endif

      /* If we've found an execute file, return it.  We have to get
	 the system name, which is easy for HDB or TAYLOR.  For other
	 spool directory schemes, we have to pull it out of the X.
	 file name; this would be insecure, except that zsfind_file
	 clobbers the file name to include the real system name.  */
      if (q != NULL
	  && q->d_name[0] == 'X'
	  && q->d_name[1] == '.')
	{
	  char *zret;

#if SPOOLDIR_HDB || SPOOLDIR_SVR4 || SPOOLDIR_TAYLOR
	  *pzsystem = zbufcpy (zSsystem);
#else
	  {
	    size_t clen;

	    clen = strlen (q->d_name) - 7;
	    *pzsystem = zbufalc (clen + 1);
	    memcpy (*pzsystem, q->d_name + 2, clen);
	    (*pzsystem)[clen] = '\0';
	  }
#endif

	  zret = zsysdep_in_dir (zSdir, q->d_name);
#if DEBUG > 1
	  DEBUG_MESSAGE2 (DEBUG_SPOOLDIR,
			  "zsysdep_get_xqt: Returning %s (system %s)",
			  zret, *pzsystem);
#endif
	  return zret;
	}
	    
      /* If we've reached the end of the directory, then if we are
	 using subdirectories loop around to read the next one,
	 otherwise we are finished.  */
      if (q == NULL)
	{
	  (void) closedir (qdir);

#if SUBDIRS
	  qSxqt_dir = NULL;
	  if (! fSone_dir)
	    continue;
#endif

	  qSxqt_topdir = NULL;
	  return NULL;
	}
    }
}

/* Free up the results of an execute file scan, when we're done with
   this system.  */

/*ARGSUSED*/
void
usysdep_get_xqt_free (zsystem)
     const char *zsystem ATTRIBUTE_UNUSED;
{
  if (qSxqt_topdir != NULL)
    {
      (void) closedir (qSxqt_topdir);
      qSxqt_topdir = NULL;
    }
#if SUBDIRS
  if (qSxqt_dir != NULL)
    {
      (void) closedir (qSxqt_dir);
      qSxqt_dir = NULL;
    }
  ubuffree (zSdir);
  zSdir = NULL;
  ubuffree (zSsystem);
  zSsystem = NULL;
  fSone_dir = FALSE;
#endif
}
