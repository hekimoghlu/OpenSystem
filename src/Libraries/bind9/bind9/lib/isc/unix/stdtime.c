/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
/* $Id: stdtime.c,v 1.19 2007/06/19 23:47:18 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stddef.h>	/* NULL */
#include <stdlib.h>	/* NULL */
#include <syslog.h>

#include <sys/time.h>

#include <isc/stdtime.h>
#include <isc/util.h>

#ifndef ISC_FIX_TV_USEC
#define ISC_FIX_TV_USEC 1
#endif

#define US_PER_S 1000000

#if ISC_FIX_TV_USEC
static inline void
fix_tv_usec(struct timeval *tv) {
	isc_boolean_t fixed = ISC_FALSE;

	if (tv->tv_usec < 0) {
		fixed = ISC_TRUE;
		do {
			tv->tv_sec -= 1;
			tv->tv_usec += US_PER_S;
		} while (tv->tv_usec < 0);
	} else if (tv->tv_usec >= US_PER_S) {
		fixed = ISC_TRUE;
		do {
			tv->tv_sec += 1;
			tv->tv_usec -= US_PER_S;
		} while (tv->tv_usec >=US_PER_S);
	}
	/*
	 * Call syslog directly as we are called from the logging functions.
	 */
	if (fixed)
		(void)syslog(LOG_ERR, "gettimeofday returned bad tv_usec: corrected");
}
#endif

void
isc_stdtime_get(isc_stdtime_t *t) {
	struct timeval tv;

	/*
	 * Set 't' to the number of seconds since 00:00:00 UTC, January 1,
	 * 1970.
	 */

	REQUIRE(t != NULL);

	RUNTIME_CHECK(gettimeofday(&tv, NULL) != -1);

#if ISC_FIX_TV_USEC
	fix_tv_usec(&tv);
	INSIST(tv.tv_usec >= 0);
#else
	INSIST(tv.tv_usec >= 0 && tv.tv_usec < US_PER_S);
#endif

	*t = (unsigned int)tv.tv_sec;
}
