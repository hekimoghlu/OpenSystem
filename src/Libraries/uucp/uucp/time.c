/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
const char time_rcsid[] = "$Id: time.c,v 1.22 2002/03/05 19:10:41 ian Rel $";
#endif

#include <ctype.h>

#if TM_IN_SYS_TIME
#include <sys/time.h>
#else
#include <time.h>
#endif

#include "uudefs.h"
#include "uuconf.h"

/* External functions.  */
#ifndef time
extern time_t time ();
#endif
#ifndef localtime
extern struct tm *localtime ();
#endif

/* See if the current time matches a time span.  If it does, return
   TRUE, set *pival to the value for the matching span, and set
   *pcretry to the retry for the matching span.  Otherwise return
   FALSE.  */

boolean
ftimespan_match (qspan, pival, pcretry)
     const struct uuconf_timespan *qspan;
     long *pival;
     int *pcretry;
{
  time_t inow;
  struct tm *qtm;
  int itm;
  const struct uuconf_timespan *q;

  if (qspan == NULL)
    return FALSE;

  time (&inow);
  qtm = localtime (&inow);

  /* Get the number of minutes since Sunday for the time.  */
  itm = qtm->tm_wday * 24 * 60 + qtm->tm_hour * 60 + qtm->tm_min;

  for (q = qspan; q != NULL; q = q->uuconf_qnext)
    {
      if (q->uuconf_istart <= itm && itm <= q->uuconf_iend)
	{
	  if (pival != NULL)
	    *pival = q->uuconf_ival;
	  if (pcretry != NULL)
	    *pcretry = q->uuconf_cretry;
	  return TRUE;
	}
    }

  return FALSE;
}

/* Determine the maximum size that may ever be transferred, according
   to a timesize span.  This returns -1 if there is no limit.  */

long
cmax_size_ever (qtimesize)
     const struct uuconf_timespan *qtimesize;
{
  long imax;
  const struct uuconf_timespan *q;

  if (qtimesize == NULL)
    return -1;

  /* Look through the list of spans.  If there is any gap larger than
     1 hour, we assume there are no restrictions.  Otherwise we keep
     track of the largest value we see.  I picked 1 hour arbitrarily,
     on the theory that a 1 hour span to transfer large files might
     actually occur, and is probably not an accident.  */
  if (qtimesize->uuconf_istart >= 60)
    return -1;

  imax = -1;

  for (q = qtimesize; q != NULL; q = q->uuconf_qnext)
    {
      if (q->uuconf_qnext == NULL)
	{
	  if (q->uuconf_iend <= 6 * 24 * 60 + 23 * 60)
	    return -1;
	}
      else
	{
	  if (q->uuconf_iend + 60 <= q->uuconf_qnext->uuconf_istart)
	    return -1;
	}

      if (imax < q->uuconf_ival)
	imax = q->uuconf_ival;
    }

  return imax;
}
