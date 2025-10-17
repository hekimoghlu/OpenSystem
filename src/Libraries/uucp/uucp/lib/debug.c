/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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

#include <ctype.h>

#include "uudefs.h"

#if DEBUG > 1

/* The debugging level.  */
int iDebug;

/* Parse a debugging string.  This may be a simple number, which sets
   the given number of bits in iDebug, or it may be a series of single
   letters.  */

static const char * const azDebug_names[] = DEBUG_NAMES;

int
idebug_parse (z)
     const char *z;
{
  char *zend;
  int i, iret;
  char *zcopy, *ztok;

  if (strncasecmp (z, DEBUG_NONE, sizeof DEBUG_NONE - 1) == 0)
    return 0;

  i = (int) strtol ((char *) z, &zend, 0);
  if (*zend == '\0')
    {
      if (i > 15)
	i = 15;
      else if (i < 0)
	i = 0;
      return (1 << i) - 1;
    }

  zcopy = zbufcpy (z);

  iret = 0;

  for (ztok = strtok (zcopy, ", \t");
       ztok != NULL;
       ztok = strtok ((char *) NULL, ", \t"))
    {
      if (strcasecmp (ztok, "all") == 0)
	{
	  iret = DEBUG_MAX;
	  break;
	}
      for (i = 0; azDebug_names[i] != NULL; i++)
	{
	  if (strncasecmp (ztok, azDebug_names[i],
			   strlen (azDebug_names[i])) == 0)
	    {
	      iret |= 1 << i;
	      break;
	    }
	}
      if (azDebug_names[i] == NULL)
	ulog (LOG_ERROR, "Unrecognized debugging option \"%s\"",
	      ztok);
    }

  ubuffree (zcopy);

  return iret;
}

#endif /* DEBUG > 1 */

/* A debugging routine used when displaying buffers.  */

size_t
cdebug_char (z, ichar)
     char *z;
     int ichar;
{
  char b;

  if (isprint (BUCHAR (ichar))
      && ichar != '\"'
      && ichar != '\\')
    {
      *z++ = (char) ichar;
      *z = '\0';
      return 1;
    }

  *z++ = '\\';

  switch (ichar)
    {
    case '\n':
      b = 'n';
      break;
    case '\r':
      b = 'r';
      break;
    case '\"':
      b = '\"';
      break;
    case '\\':
      b = '\\';
      break;
    default:
      sprintf (z, "%03o", (unsigned int) BUCHAR (ichar));
      return strlen (z) + 1;
    }

  *z++ = b;
  *z = '\0';
  return 2;
}      

#if DEBUG > 1

/* Display a buffer when debugging.  */

void
udebug_buffer (zhdr, zbuf, clen)
     const char *zhdr;
     const char *zbuf;
     size_t clen;
{
  char *z, *zalc;
  size_t i;

  zalc = zbufalc (clen * 4 + 1);

  z = zalc;
  for (i = 0; i < clen && i < 80; i++)
    z += cdebug_char (z, zbuf[i]);
  if (i < clen)
    {
      *z++ = '.';
      *z++ = '.';
      *z++ = '.';
    }
  *z = '\0';

  ulog (LOG_DEBUG, "%s %lu \"%s\"", zhdr, (unsigned long) clen, zalc);

  ubuffree (zalc);
}

#endif
