/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
/* $Id: ttl.h,v 1.19 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DNS_TTL_H
#define DNS_TTL_H 1

/*! \file dns/ttl.h */

/***
 ***	Imports
 ***/

#include <isc/lang.h>
#include <isc/types.h>

ISC_LANG_BEGINDECLS

/***
 ***	Functions
 ***/

isc_result_t
dns_ttl_totext(isc_uint32_t src, isc_boolean_t verbose,
	       isc_buffer_t *target);
/*%<
 * Output a TTL or other time interval in a human-readable form.
 * The time interval is given as a count of seconds in 'src'.
 * The text representation is appended to 'target'.
 *
 * If 'verbose' is ISC_FALSE, use the terse BIND 8 style, like "1w2d3h4m5s".
 *
 * If 'verbose' is ISC_TRUE, use a verbose style like the SOA comments
 * in "dig", like "1 week 2 days 3 hours 4 minutes 5 seconds".
 *
 * Returns:
 * \li	ISC_R_SUCCESS
 * \li	ISC_R_NOSPACE
 */

isc_result_t
dns_counter_fromtext(isc_textregion_t *source, isc_uint32_t *ttl);
/*%<
 * Converts a counter from either a plain number or a BIND 8 style value.
 *
 * Returns:
 *\li	ISC_R_SUCCESS
 *\li	DNS_R_SYNTAX
 */

isc_result_t
dns_ttl_fromtext(isc_textregion_t *source, isc_uint32_t *ttl);
/*%<
 * Converts a ttl from either a plain number or a BIND 8 style value.
 *
 * Returns:
 *\li	ISC_R_SUCCESS
 *\li	DNS_R_BADTTL
 */

ISC_LANG_ENDDECLS

#endif /* DNS_TTL_H */
