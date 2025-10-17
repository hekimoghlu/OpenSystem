/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
const char _uuconf_callin_rcsid[] = "$Id: callin.c,v 1.14 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

static int ipcheck P((pointer pglobal, int argc, char **argv,
		      pointer pvar, pointer pinfo));

struct sinfo
{
  int (*pcmpfn) P((int, pointer, const char *));
  pointer pinfo;
  boolean ffound;
  boolean fmatched;
};

/* Check a login name and password against the UUCP password file.
   This looks at the Taylor UUCP password file, but will work even if
   uuconf_taylor_init was not called.  It accepts either spaces or
   colons as field delimiters.  */

int
uuconf_callin (pglobal, pcmpfn, pinfo)
     pointer pglobal;
     int (*pcmpfn) P((int, pointer, const char *));
     pointer pinfo;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  int iret;
  char **pz;
  struct uuconf_cmdtab as[1];
  struct sinfo s;
  char *zline;
  size_t cline;

  /* If we have no password file names, fill in the default name.  */
  if (qglobal->qprocess->pzpwdfiles == NULL)
    {
      char ab[sizeof NEWCONFIGLIB + sizeof PASSWDFILE - 1];

      memcpy ((pointer) ab, (pointer) NEWCONFIGLIB,
	      sizeof NEWCONFIGLIB - 1);
      memcpy ((pointer) (ab + sizeof NEWCONFIGLIB - 1), (pointer) PASSWDFILE,
	      sizeof PASSWDFILE);
      iret = _uuconf_iadd_string (qglobal, ab, TRUE, FALSE,
				  &qglobal->qprocess->pzpwdfiles,
				  qglobal->pblock);
      if (iret != UUCONF_SUCCESS)
	return iret;
    }

  as[0].uuconf_zcmd = NULL;

  s.pcmpfn = pcmpfn;
  s.pinfo = pinfo;
  s.ffound = FALSE;
  s.fmatched = FALSE;

  zline = NULL;
  cline = 0;

  iret = UUCONF_SUCCESS;

  for (pz = qglobal->qprocess->pzpwdfiles; *pz != NULL; pz++)
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

      qglobal->ilineno = 0;

      iret = UUCONF_SUCCESS;

      while (getline (&zline, &cline, e) > 0)
	{
	  char *z0, *z1;

	  ++qglobal->ilineno;

	  /* We have a few hacks to make Unix style passwd files work.
	     1) We turn the first two colon characters into spaces.
	     2) If the colon characters are adjacent, we assume there
	        is no password, and we skip the entry.
	     3) If the password between colon characters contains a
	        space, we assume that it has been disabled, and we
		skip the entry.  */
	  z0 = strchr (zline, ':');
	  if (z0 != NULL)
	    {
	      *z0 = ' ';
	      z1 = strchr (z0, ':');
	      if (z1 != NULL)
		{
		  if (z1 - z0 == 1)
		    continue;
		  *z1 = '\0';
		  if (strchr (z0 + 1, ' ') != NULL)
		    continue;
		}
	    }		  
	  iret = uuconf_cmd_line (pglobal, zline, as, (pointer) &s,
				  ipcheck, 0, (pointer) NULL);
	  if ((iret & UUCONF_CMDTABRET_EXIT) != 0)
	    {
	      iret &=~ UUCONF_CMDTABRET_EXIT;
	      if (iret != UUCONF_SUCCESS)
		iret |= UUCONF_ERROR_LINENO;
	      break;
	    }

	  iret = UUCONF_SUCCESS;
	}

      (void) fclose (e);

      if (iret != UUCONF_SUCCESS || s.ffound)
	break;
    }

  if (zline != NULL)
    free ((pointer) zline);

  if (iret != UUCONF_SUCCESS)
    {
      qglobal->zfilename = *pz;
      iret |= UUCONF_ERROR_FILENAME;
    }
  else if (! s.ffound || ! s.fmatched)
    iret = UUCONF_NOT_FOUND;

  return iret;
}

/* This is called on each line of the file.  It checks to see if the
   login name from the file is the one we are looking for.  If it is,
   it sets ffound, and then sets fmatched according to whether the
   password matches or not.  */

static int
ipcheck (pglobal, argc, argv, pvar, pinfo)
     pointer pglobal ATTRIBUTE_UNUSED;
     int argc;
     char **argv;
     pointer pvar ATTRIBUTE_UNUSED;
     pointer pinfo;
{
  struct sinfo *q = (struct sinfo *) pinfo;

  if (argc != 2)
    return UUCONF_SYNTAX_ERROR | UUCONF_CMDTABRET_EXIT;

  if (! (*q->pcmpfn) (0, q->pinfo, argv[0]))
    return UUCONF_CMDTABRET_CONTINUE;

  q->ffound = TRUE;
  q->fmatched = (*q->pcmpfn) (1, q->pinfo, argv[1]) != 0;

  return UUCONF_CMDTABRET_EXIT;
}
