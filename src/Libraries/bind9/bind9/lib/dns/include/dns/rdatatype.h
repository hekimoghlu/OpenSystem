/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
/* $Id: rdatatype.h,v 1.26 2008/09/25 04:02:39 tbox Exp $ */

#ifndef DNS_RDATATYPE_H
#define DNS_RDATATYPE_H 1

/*! \file dns/rdatatype.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_rdatatype_fromtext(dns_rdatatype_t *typep, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a DNS rdata type.
 *
 * Requires:
 *\li	'typep' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	ISC_R_SUCCESS			on success
 *\li	DNS_R_UNKNOWN			type is unknown
 */

isc_result_t
dns_rdatatype_totext(dns_rdatatype_t type, isc_buffer_t *target);
/*%<
 * Put a textual representation of type 'type' into 'target'.
 *
 * Requires:
 *\li	'type' is a valid type.
 *
 *\li	'target' is a valid text buffer.
 *
 * Ensures,
 *	if the result is success:
 *\li		The used space in 'target' is updated.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#ISC_R_NOSPACE			target buffer is too small
 */

void
dns_rdatatype_format(dns_rdatatype_t rdtype,
		     char *array, unsigned int size);
/*%<
 * Format a human-readable representation of the type 'rdtype'
 * into the character array 'array', which is of size 'size'.
 * The resulting string is guaranteed to be null-terminated.
 */

#define DNS_RDATATYPE_FORMATSIZE sizeof("NSEC3PARAM")

/*%<
 * Minimum size of array to pass to dns_rdatatype_format().
 * May need to be adjusted if a new RR type with a very long
 * name is defined.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_RDATATYPE_H */
