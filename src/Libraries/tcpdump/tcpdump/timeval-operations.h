/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef netdissect_timeval_operations_h
#define netdissect_timeval_operations_h

/* Operations on timevals. */

#define ND_MICRO_PER_SEC 1000000
#define ND_NANO_PER_SEC 1000000000

#define netdissect_timevalclear(tvp) ((tvp)->tv_sec = (tvp)->tv_usec = 0)

#define netdissect_timevalisset(tvp) ((tvp)->tv_sec || (tvp)->tv_usec)

#define netdissect_timevalcmp(tvp, uvp, cmp)   \
	(((tvp)->tv_sec == (uvp)->tv_sec) ?    \
	 ((tvp)->tv_usec cmp (uvp)->tv_usec) : \
	 ((tvp)->tv_sec cmp (uvp)->tv_sec))

#define netdissect_timevaladd(tvp, uvp, vvp, nano_prec)           \
	do {                                                      \
		(vvp)->tv_sec = (tvp)->tv_sec + (uvp)->tv_sec;    \
		(vvp)->tv_usec = (tvp)->tv_usec + (uvp)->tv_usec; \
		if (nano_prec) {                                  \
			if ((vvp)->tv_usec >= ND_NANO_PER_SEC) {  \
				(vvp)->tv_sec++;                  \
				(vvp)->tv_usec -= ND_NANO_PER_SEC; \
			}                                         \
		} else {                                          \
			if ((vvp)->tv_usec >= ND_MICRO_PER_SEC) { \
				(vvp)->tv_sec++;                  \
				(vvp)->tv_usec -= ND_MICRO_PER_SEC; \
			}                                         \
		}                                                 \
	} while (0)

#define netdissect_timevalsub(tvp, uvp, vvp, nano_prec)            \
	do {                                                       \
		(vvp)->tv_sec = (tvp)->tv_sec - (uvp)->tv_sec;     \
		(vvp)->tv_usec = (tvp)->tv_usec - (uvp)->tv_usec;  \
		if ((vvp)->tv_usec < 0) {                          \
		    (vvp)->tv_sec--;                               \
		    (vvp)->tv_usec += (nano_prec ? ND_NANO_PER_SEC : \
				       ND_MICRO_PER_SEC);          \
		}                                                  \
	} while (0)

#endif /* netdissect_timeval_operations_h */
