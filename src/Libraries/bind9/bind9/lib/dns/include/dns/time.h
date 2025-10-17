/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
/* $Id: time.h,v 1.19 2012/01/27 23:46:58 tbox Exp $ */

#ifndef DNS_TIME_H
#define DNS_TIME_H 1

/*! \file dns/time.h */

/***
 ***	Imports
 ***/

#include <isc/buffer.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

/***
 ***	Functions
 ***/

isc_result_t
dns_time64_fromtext(const char *source, isc_int64_t *target);
/*%<
 * Convert a date and time in YYYYMMDDHHMMSS text format at 'source'
 * into to a 64-bit count of seconds since Jan 1 1970 0:00 GMT.
 * Store the count at 'target'.
 */

isc_result_t
dns_time32_fromtext(const char *source, isc_uint32_t *target);
/*%<
 * Like dns_time64_fromtext, but returns the second count modulo 2^32
 * as per RFC2535.
 */


isc_result_t
dns_time64_totext(isc_int64_t value, isc_buffer_t *target);
/*%<
 * Convert a 64-bit count of seconds since Jan 1 1970 0:00 GMT into
 * a YYYYMMDDHHMMSS text representation and append it to 'target'.
 */

isc_result_t
dns_time32_totext(isc_uint32_t value, isc_buffer_t *target);
/*%<
 * Like dns_time64_totext, but for a 32-bit cyclic time value.
 * Of those dates whose counts of seconds since Jan 1 1970 0:00 GMT
 * are congruent with 'value' modulo 2^32, the one closest to the
 * current date is chosen.
 */

isc_int64_t
dns_time64_from32(isc_uint32_t value);
/*%<
 * Covert a 32-bit cyclic time value into a 64 bit time stamp.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_TIME_H */
