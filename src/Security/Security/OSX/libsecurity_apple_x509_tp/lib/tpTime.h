/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
/*
 * tpTime.h - cert related time functions
 *
 */

#ifndef	_TP_TIME_H_
#define _TP_TIME_H_

#include <time.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/cssmtype.h>

#ifdef	__cplusplus
extern "C" {
#endif

/* lengths of time strings without trailing NULL */
#define UTC_TIME_NOSEC_LEN			11
#define UTC_TIME_STRLEN				13
#define CSSM_TIME_STRLEN			14		/* no trailing 'Z' */
#define GENERALIZED_TIME_STRLEN		15
#define LOCALIZED_UTC_TIME_STRLEN	17
#define LOCALIZED_TIME_STRLEN		19

/*
 * Given a string containing either a UTC-style or "generalized time"
 * time string, convert to a CFDateRef. Returns nonzero on
 * error.
 */
extern int timeStringToCfDate(
	const char			*str,
	unsigned			len,
	CFDateRef			*cfDate);

/*
 * Compare two times. Assumes they're both in GMT. Returns:
 * -1 if t1 <  t2
 *  0 if t1 == t2
 *  1 if t1 >  t2
 */
extern int compareTimes(
	CFDateRef 	t1,
	CFDateRef 	t2);

/*
 * Create a time string, in either UTC (2-digit) or or Generalized (4-digit)
 * year format. Caller mallocs the output string whose length is at least
 * (UTC_TIME_STRLEN+1), (GENERALIZED_TIME_STRLEN+1), or (CSSM_TIME_STRLEN+1)
 * respectively. Caller must hold tpTimeLock.
 */
typedef enum {
	TP_TIME_UTC = 0,
	TP_TIME_GEN = 1,
	TP_TIME_CSSM = 2
} TpTimeSpec;

void timeAtNowPlus(unsigned secFromNow,
	TpTimeSpec timeSpec,
	char *outStr,
	size_t outStrSize);

/*
 * Convert a time string, which can be in any of three forms (UTC,
 * generalized, or CSSM_TIMESTRING) into a CSSM_TIMESTRING. Caller
 * mallocs the result, which must be at least (CSSM_TIME_STRLEN+1) bytes.
 * Returns nonzero if incoming time string is badly formed.
 */
int tpTimeToCssmTimestring(
	const char 	*inStr,			// not necessarily NULL terminated
	unsigned	inStrLen,		// not including possible NULL
	char 		*outTime);		// caller mallocs

#ifdef	__cplusplus
}
#endif

#endif	/* _TP_TIME_H_*/
