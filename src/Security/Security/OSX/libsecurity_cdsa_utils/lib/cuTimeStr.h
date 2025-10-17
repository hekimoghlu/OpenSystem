/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
 * cuTimeStr.h = Time string utilities.
 */

#ifndef	_TIME_STR_H_
#define _TIME_STR_H_

#include <time.h>
#include <Security/x509defs.h>

#define UTC_TIME_NOSEC_LEN			11
#define UTC_TIME_STRLEN				13
#define CSSM_TIME_STRLEN			14		/* no trailing 'Z' */
#define GENERALIZED_TIME_STRLEN		15

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Given a string containing either a UTC-style or "generalized time"
 * time string, convert to a struct tm (in GMT/UTC). Returns nonzero on
 * error.
 */
int cuTimeStringToTm(
	const char			*str,
	unsigned			len,
	struct tm			*tmp);

typedef enum {
	CU_TIME_UTC = 0,
	CU_TIME_LEGACY = 1, /* do not use; see note below */
	CU_TIME_GEN = 2,
	CU_TIME_CSSM = 3
} timeSpec;

/* TIME_UTC was formerly defined in this header with a value of 0. However,
 * a newer version of <time.h> may define it with a value of 1. Clients which
 * specify the TIME_UTC constant for a timeSpec parameter will get a value
 * of 0 under the old header, and a value of 1 under the newer header.
 * The latter conflicts with the old definition of TIME_CSSM. To resolve this,
 * we now treat 1 as a legacy value which maps to CU_TIME_UTC if TIME_UTC=1,
 * otherwise to CU_TIME_CSSM.
 *
 * Important: any code which specifies the legacy TIME_CSSM constant must be
 * recompiled against this header to ensure the correct timeSpec value is used.
 */
#ifndef TIME_UTC
#define TIME_UTC CU_TIME_UTC /* time.h has not defined TIME_UTC */
#endif
#ifndef TIME_GEN
#define TIME_GEN CU_TIME_GEN
#endif
#ifndef TIME_CSSM
#define TIME_CSSM CU_TIME_CSSM
#endif

/*
 * Return an APP_MALLOCd time string, specified format and time relative
 * to 'now' in seconds.
 */
char *cuTimeAtNowPlus(
	int 				secFromNow,
	timeSpec 			spec);

/*
 * Convert a CSSM_X509_TIME, which can be in any of three forms (UTC,
 * generalized, or CSSM_TIMESTRING) into a CSSM_TIMESTRING. Caller
 * must free() the result. Returns NULL if x509time is badly formed.
 */
char *cuX509TimeToCssmTimestring(
	const CSSM_X509_TIME 	*x509Time,
	unsigned				*rtnLen);		// for caller's convenience

#ifdef	__cplusplus
}
#endif

#endif	/* _TIME_STR_H_ */
