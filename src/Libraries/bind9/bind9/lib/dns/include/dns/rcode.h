/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
/* $Id: rcode.h,v 1.21 2008/09/25 04:02:39 tbox Exp $ */

#ifndef DNS_RCODE_H
#define DNS_RCODE_H 1

/*! \file dns/rcode.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t dns_rcode_fromtext(dns_rcode_t *rcodep, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a DNS error value.
 *
 * Requires:
 *\li	'rcodep' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#DNS_R_UNKNOWN			type is unknown
 */

isc_result_t dns_rcode_totext(dns_rcode_t rcode, isc_buffer_t *target);
/*%<
 * Put a textual representation of error 'rcode' into 'target'.
 *
 * Requires:
 *\li	'rcode' is a valid rcode.
 *
 *\li	'target' is a valid text buffer.
 *
 * Ensures:
 *\li	If the result is success:
 *		The used space in 'target' is updated.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#ISC_R_NOSPACE			target buffer is too small
 */

isc_result_t dns_tsigrcode_fromtext(dns_rcode_t *rcodep,
				    isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a TSIG/TKEY error value.
 *
 * Requires:
 *\li	'rcodep' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#DNS_R_UNKNOWN			type is unknown
 */

isc_result_t dns_tsigrcode_totext(dns_rcode_t rcode, isc_buffer_t *target);
/*%<
 * Put a textual representation of TSIG/TKEY error 'rcode' into 'target'.
 *
 * Requires:
 *\li	'rcode' is a valid TSIG/TKEY error code.
 *
 *\li	'target' is a valid text buffer.
 *
 * Ensures:
 *\li	If the result is success:
 *		The used space in 'target' is updated.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#ISC_R_NOSPACE			target buffer is too small
 */

isc_result_t
dns_hashalg_fromtext(unsigned char *hashalg, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a has algorithm value.
 *
 * Requires:
 *\li	'hashalg' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#DNS_R_UNKNOWN			type is unknown
 */

ISC_LANG_ENDDECLS

#endif /* DNS_RCODE_H */
