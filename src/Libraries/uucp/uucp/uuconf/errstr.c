/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
const char _uuconf_errstr_rcsid[] = "$Id: errstr.c,v 1.6 2002/03/05 19:10:42 ian Rel $";
#endif

static char *zeprint_num P((char *zbuf, size_t cbuf, int ival));

/* Return an error string for a uuconf error.  This does not return a
   uuconf error code, but instead returns the total buffer length.  */

int
uuconf_error_string (pglobal, ierr, zbuf, cbuf)
     pointer pglobal;
     int ierr;
     char *zbuf;
     size_t cbuf;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;
  const char *zfile;
  size_t cfile;
  const char *zlineno;
  char ablineno[100];
  size_t clineno;
  const char *zmsg;
  char abmsg[100];
  size_t cmsg;
  const char *zerrno;
  size_t cerrno;
  size_t cret;
  size_t ccopy;

  /* The format of the message is

     filename:lineno: message: errno

     If there is no filename, the trailing colon is not output.  If
     there is no linenumber, the trailing colon is not output.  If
     there is no filename, the linenumber is not output, and neither
     is the space before message.  If there is no errno, the
     preceeding colon and space are not output.  */

  /* Get the filename to put in the error message, if any.  */
  if ((ierr & UUCONF_ERROR_FILENAME) == 0
      || qglobal == NULL
      || qglobal->zfilename == NULL)
    {
      zfile = "";
      cfile = 0;
    }
  else
    {
      zfile = qglobal->zfilename;
      cfile = strlen (zfile) + 1;
    }

  /* Get the line number to put in the error message, if any.  */
  if (cfile == 0
      || (ierr & UUCONF_ERROR_LINENO) == 0
      || qglobal == NULL
      || qglobal->ilineno <= 0)
    {
      zlineno = "";
      clineno = 0;
    }
  else
    {
      zlineno = zeprint_num (ablineno, sizeof ablineno, qglobal->ilineno);
      clineno = strlen (zlineno) + 1;
    }

  /* Get the main message.  */
  switch (UUCONF_ERROR_VALUE (ierr))
    {
    case UUCONF_SUCCESS:
      zmsg = "no error";
      break;
    case UUCONF_NOT_FOUND:
      zmsg = "not found";
      break;
    case UUCONF_FOPEN_FAILED:
      zmsg = "fopen";
      break;
    case UUCONF_FSEEK_FAILED:
      zmsg = "fseek";
      break;
    case UUCONF_MALLOC_FAILED:
      zmsg = "malloc";
      break;
    case UUCONF_SYNTAX_ERROR:
      zmsg = "syntax error";
      break;
    default:
      zmsg = zeprint_num (abmsg, sizeof abmsg, UUCONF_ERROR_VALUE (ierr));
      zmsg -= sizeof "error " - 1;
      memcpy ((pointer) zmsg, (pointer) "error ", sizeof "error " - 1);
      break;
    }

  cmsg = strlen (zmsg);
  if (cfile > 0)
    ++cmsg;

  /* Get the errno string.  Note that strerror is not necessarily
     reentrant.  */
  if ((ierr & UUCONF_ERROR_ERRNO) == 0
      || qglobal == NULL)
    {
      zerrno = "";
      cerrno = 0;
    }
  else
    {
      zerrno = strerror (qglobal->ierrno);
      cerrno = strlen (zerrno) + 2;
    }

  cret = cfile + clineno + cmsg + cerrno + 1;

  if (cbuf == 0)
    return cret;

  /* Leave room for the null byte.  */
  --cbuf;

  if (cfile > 0)
    {
      ccopy = cfile - 1;
      if (ccopy > cbuf)
	ccopy = cbuf;
      memcpy ((pointer) zbuf, (pointer) zfile, ccopy);
      zbuf += ccopy;
      cbuf -= ccopy;
      if (cbuf > 0)
	{
	  *zbuf++ = ':';
	  --cbuf;
	}
    }

  if (clineno > 0)
    {
      ccopy = clineno - 1;
      if (ccopy > cbuf)
	ccopy = cbuf;
      memcpy ((pointer) zbuf, (pointer) zlineno, ccopy);
      zbuf += ccopy;
      cbuf -= ccopy;
      if (cbuf > 0)
	{
	  *zbuf++ = ':';
	  --cbuf;
	}
    }
      
  if (cbuf > 0 && cfile > 0)
    {
      *zbuf++ = ' ';
      --cbuf;
      --cmsg;
    }
  ccopy = cmsg;
  if (ccopy > cbuf)
    ccopy = cbuf;
  memcpy ((pointer) zbuf, (pointer) zmsg, ccopy);
  zbuf += ccopy;
  cbuf -= ccopy;

  if (cerrno > 0)
    {
      if (cbuf > 0)
	{
	  *zbuf++ = ':';
	  --cbuf;
	}
      if (cbuf > 0)
	{
	  *zbuf++ = ' ';
	  --cbuf;
	}
      ccopy = cerrno - 2;
      if (ccopy > cbuf)
	ccopy = cbuf;
      memcpy ((pointer) zbuf, (pointer) zerrno, ccopy);
      zbuf += ccopy;
      cbuf -= ccopy;
    }

  *zbuf = '\0';

  return cret;
}

/* Turn a number into a string.  This should really call sprintf, but
   since nothing else in the uuconf library calls any print routine,
   it's more interesting to not call it here either.  */

static char *
zeprint_num (ab, c, i)
     char *ab;
     size_t c;
     register int i;
{
  register char *z;

  z = ab + c;
  *--z = '\0';
  do
    {
      *--z = i % 10 + '0';
      i /= 10;
    }
  while (i != 0);

  return z;
}
