/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
/* $Id$ */

#ifndef DNS_DSDIGEST_H
#define DNS_DSDIGEST_H 1

/*! \file dns/dsdigest.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_dsdigest_fromtext(dns_dsdigest_t *dsdigestp, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a DS/DLV digest type value.
 * The text may contain either a mnemonic digest name or a decimal
 * digest number.
 *
 * Requires:
 *\li	'dsdigestp' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	ISC_R_SUCCESS			on success
 *\li	ISC_R_RANGE			numeric type is out of range
 *\li	DNS_R_UNKNOWN			mnemonic type is unknown
 */

isc_result_t
dns_dsdigest_totext(dns_dsdigest_t dsdigest, isc_buffer_t *target);
/*%<
 * Put a textual representation of the DS/DLV digest type 'dsdigest'
 * into 'target'.
 *
 * Requires:
 *\li	'dsdigest' is a valid dsdigest.
 *
 *\li	'target' is a valid text buffer.
 *
 * Ensures,
 *	if the result is success:
 *\li		The used space in 'target' is updated.
 *
 * Returns:
 *\li	ISC_R_SUCCESS			on success
 *\li	ISC_R_NOSPACE			target buffer is too small
 */

#define DNS_DSDIGEST_FORMATSIZE 20
void
dns_dsdigest_format(dns_dsdigest_t typ, char *cp, unsigned int size);
/*%<
 * Wrapper for dns_dsdigest_totext(), writing text into 'cp'
 */

ISC_LANG_ENDDECLS

#endif /* DNS_DSDIGEST_H */
