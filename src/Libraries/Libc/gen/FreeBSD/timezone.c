/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)timezone.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/timezone.c,v 1.6 2007/01/09 00:27:55 imp Exp $");

#include <sys/types.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TZ_MAX_CHARS 255

char *_tztab();

/*
 * timezone --
 *	The arguments are the number of minutes of time you are westward
 *	from Greenwich and whether DST is in effect.  It returns a string
 *	giving the name of the local timezone.  Should be replaced, in the
 *	application code, by a call to localtime.
 */

static char	czone[TZ_MAX_CHARS];		/* space for zone name */

char *
timezone(zone, dst)
	int	zone,
		dst;
{
	char	*beg,
			*end;

	if ( (beg = getenv("TZNAME")) ) {	/* set in environment */
		if ( (end = index(beg, ',')) ) {/* "PST,PDT" */
			if (dst)
				return(++end);
			*end = '\0';
			(void)strncpy(czone,beg,sizeof(czone) - 1);
			czone[sizeof(czone) - 1] = '\0';
			*end = ',';
			return(czone);
		}
		return(beg);
	}
	return(_tztab(zone,dst));	/* default: table or created zone */
}

static struct zone {
	int	offset;
	char	*stdzone;
	char	*dlzone;
} zonetab[] = {
	{-1*60,	"MET",	"MET DST"},	/* Middle European */
	{-2*60,	"EET",	"EET DST"},	/* Eastern European */
	{4*60,	"AST",	"ADT"},		/* Atlantic */
	{5*60,	"EST",	"EDT"},		/* Eastern */
	{6*60,	"CST",	"CDT"},		/* Central */
	{7*60,	"MST",	"MDT"},		/* Mountain */
	{8*60,	"PST",	"PDT"},		/* Pacific */
#ifdef notdef
	/* there's no way to distinguish this from WET */
	{0,	"GMT",	0},		/* Greenwich */
#endif
	{0*60,	"WET",	"WET DST"},	/* Western European */
	{-10*60,"EST",	"EST"},		/* Aust: Eastern */
     {-10*60+30,"CST",	"CST"},		/* Aust: Central */
	{-8*60,	"WST",	0},		/* Aust: Western */
	{-1}
};

/*
 * _tztab --
 *	check static tables or create a new zone name; broken out so that
 *	we can make a guess as to what the zone is if the standard tables
 *	aren't in place in /etc.  DO NOT USE THIS ROUTINE OUTSIDE OF THE
 *	STANDARD LIBRARY.
 */
char *
_tztab(zone,dst)
	int	zone;
	int	dst;
{
	struct zone	*zp;
	char	sign;

	for (zp = zonetab; zp->offset != -1;++zp)	/* static tables */
		if (zp->offset == zone) {
			if (dst && zp->dlzone)
				return(zp->dlzone);
			if (!dst && zp->stdzone)
				return(zp->stdzone);
		}

	if (zone < 0) {					/* create one */
		zone = -zone;
		sign = '+';
	}
	else
		sign = '-';
	(void)snprintf(czone, sizeof(czone),
	    "GMT%c%d:%02d",sign,zone / 60,zone % 60);
	return(czone);
}
#pragma clang diagnostic pop
