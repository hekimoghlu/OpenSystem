/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
/* $Id: rdataclass.h,v 1.24 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DNS_RDATACLASS_H
#define DNS_RDATACLASS_H 1

/*! \file dns/rdataclass.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_rdataclass_fromtext(dns_rdataclass_t *classp, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a DNS class.
 *
 * Requires:
 *\li	'classp' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	#ISC_R_SUCCESS			on success
 *\li	#DNS_R_UNKNOWN			class is unknown
 */

isc_result_t
dns_rdataclass_totext(dns_rdataclass_t rdclass, isc_buffer_t *target);
/*%<
 * Put a textual representation of class 'rdclass' into 'target'.
 *
 * Requires:
 *\li	'rdclass' is a valid class.
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
dns_rdataclass_format(dns_rdataclass_t rdclass,
		      char *array, unsigned int size);
/*%<
 * Format a human-readable representation of the class 'rdclass'
 * into the character array 'array', which is of size 'size'.
 * The resulting string is guaranteed to be null-terminated.
 */

#define DNS_RDATACLASS_FORMATSIZE sizeof("CLASS65535")
/*%<
 * Minimum size of array to pass to dns_rdataclass_format().
 */

ISC_LANG_ENDDECLS

#endif /* DNS_RDATACLASS_H */
