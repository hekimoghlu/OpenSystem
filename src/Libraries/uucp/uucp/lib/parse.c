/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
const char parse_rcsid[] = "$Id: parse.c,v 1.11 2002/03/05 19:10:42 ian Rel $";
#endif

#include "uudefs.h"

/* Local functions.  */

static void ulunquote_cmd P((struct scmd *qcmd));

/* Parse a UUCP command string into an scmd structure.  This is called
   by the 'g' protocol and the UNIX command file reading routines.  It
   destroys the string it is passed, and the scmd string pointers are
   left pointing into it.  For the convenience of the Unix work file
   routines, it will parse "P" into a simple 'P' command (representing
   a poll file).  If 'q' appears in the options, it will unquote all
   the relevant strings.  It returns TRUE if the string is
   successfully parsed, FALSE otherwise.  */

boolean
fparse_cmd (zcmd, qcmd)
     char *zcmd;
     struct scmd *qcmd;
{
  char *z, *zend;

  z = strtok (zcmd, " \t\n");
  if (z == NULL)
    return FALSE;

  qcmd->bcmd = *z;
  if (qcmd->bcmd != 'S'
      && qcmd->bcmd != 'R'
      && qcmd->bcmd != 'X'
      && qcmd->bcmd != 'E'
      && qcmd->bcmd != 'H'
      && qcmd->bcmd != 'P')
    return FALSE;

  qcmd->bgrade = '\0';
  qcmd->pseq = NULL;
  qcmd->zfrom = NULL;
  qcmd->zto = NULL;
  qcmd->zuser = NULL;
  qcmd->zoptions = NULL;
  qcmd->ztemp = NULL;
  qcmd->imode = 0666;
  qcmd->znotify = NULL;
  qcmd->cbytes = -1;
  qcmd->zcmd = NULL;
  qcmd->ipos = 0;

  /* Handle hangup commands specially.  If it's just "H", return
     the command 'H' to indicate a hangup request.  If it's "HY"
     return 'Y' and if it's "HN" return 'N'.  */
  if (qcmd->bcmd == 'H')
    {
      if (z[1] != '\0')
	{
	  if (z[1] == 'Y')
	    qcmd->bcmd = 'Y';
	  else if (z[1] == 'N')
	    qcmd->bcmd = 'N';
	  else
	    return FALSE;
	}

      return TRUE;
    }
  if (qcmd->bcmd == 'P')
    return TRUE;

  if (z[1] != '\0')
    return FALSE;

  z = strtok ((char *) NULL, " \t\n");
  if (z == NULL)
    return FALSE;
  qcmd->zfrom = z;

  z = strtok ((char *) NULL, " \t\n");
  if (z == NULL)
    return FALSE;
  qcmd->zto = z;
      
  z = strtok ((char *) NULL, " \t\n");
  if (z == NULL)
    return FALSE;
  qcmd->zuser = z;

  z = strtok ((char *) NULL, " \t\n");
  if (z == NULL || *z != '-')
    return FALSE;
  qcmd->zoptions = z + 1;

  if (qcmd->bcmd == 'X')
    {
      ulunquote_cmd (qcmd);
      return TRUE;
    }

  if (qcmd->bcmd == 'R')
    {
      z = strtok ((char *) NULL, " \t\n");
      if (z != NULL)
	{
	  if (strcmp (z, "dummy") != 0)
	    {
	      /* This may be the maximum number of bytes the remote
		 system wants to receive, if it using Taylor UUCP size
		 negotiation.  */
	      qcmd->cbytes = strtol (z, &zend, 0);
	      if (*zend != '\0')
		qcmd->cbytes = -1;
	    }
	  else
	    {
	      /* This is from an SVR4 system, and may include the
		 position at which to start sending the file.  The
		 next fields are the mode bits, the remote owner (?),
		 the remote temporary file name, and finally the
		 restart position.  */
	      if (strtok ((char *) NULL, " \t\n") != NULL
		  && strtok ((char *) NULL, " \t\n") != NULL
		  && strtok ((char *) NULL, " \t\n") != NULL)
		{
		  z = strtok ((char *) NULL, " \t\n");
		  if (z != NULL)
		    {
		      qcmd->ipos = strtol (z, &zend, 0);
		      if (*zend != '\0')
			qcmd->ipos = 0;
		    }
		}
	    }
	}

      ulunquote_cmd (qcmd);
      return TRUE;
    }      

  z = strtok ((char *) NULL, " \t\n");
  if (z == NULL)
    return FALSE;
  qcmd->ztemp = z;

  z = strtok ((char *) NULL, " \t\n");
  if (z == NULL)
    return FALSE;
  qcmd->imode = (int) strtol (z, &zend, 0);
  if (*zend != '\0')
    return FALSE;

  /* As a magic special case, if the mode came out as the decimal
     values 666 or 777, assume that they actually meant the octal
     values.  Most systems use a leading zero, but a few do not.
     Since both 666 and 777 are greater than the largest legal mode
     value, which is 0777 == 511, this hack does not restrict any
     legal values.  */
  if (qcmd->imode == 666)
    qcmd->imode = 0666;
  else if (qcmd->imode == 777)
    qcmd->imode = 0777;

  z = strtok ((char *) NULL, " \t\n");
  if (qcmd->bcmd == 'E' && z == NULL)
    return FALSE;
  qcmd->znotify = z;

  /* SVR4 UUCP will send the string "dummy" after the notify string
     but before the size.  I do not know when it sends anything other
     than "dummy".  Fortunately, it doesn't really hurt to not get the
     file size.  */
  if (z != NULL && strcmp (z, "dummy") == 0)
    z = strtok ((char *) NULL, " \t\n");

  if (z != NULL)
    {
      z = strtok ((char *) NULL, " \t\n");
      if (z != NULL)
	{
	  qcmd->cbytes = strtol (z, &zend, 0);
	  if (*zend != '\0')
	    qcmd->cbytes = -1;
	}
      else if (qcmd->bcmd == 'E')
	return FALSE;
      
      if (z != NULL)
	{
	  z = strtok ((char *) NULL, "");
	  if (z != NULL)
	    z[strcspn (z, "\n")] = '\0';
	  if (qcmd->bcmd == 'E' && z == NULL)
	    return FALSE;
	  qcmd->zcmd = z;
	}
    }

  ulunquote_cmd (qcmd);

  return TRUE;
}

/* If 'q' appears in the options of a command, unquote all the
   relevant strings.  */

static void
ulunquote_cmd (qcmd)
     struct scmd *qcmd;
{
  if (qcmd->zoptions == NULL || strchr (qcmd->zoptions, 'q') == NULL)
    return;

  if (qcmd->zfrom != NULL)
    (void) cescape ((char *) qcmd->zfrom);
  if (qcmd->zto != NULL)
    (void) cescape ((char *) qcmd->zto);
  if (qcmd->zuser != NULL)
    (void) cescape ((char *) qcmd->zuser);
  if (qcmd->znotify != NULL)
    (void) cescape ((char *) qcmd->znotify);
  if (qcmd->zcmd != NULL)
    (void) cescape ((char *) qcmd->zcmd);
}
