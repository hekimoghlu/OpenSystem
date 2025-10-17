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
/* $Id: secproto.h,v 1.16 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DNS_SECPROTO_H
#define DNS_SECPROTO_H 1

/*! \file dns/secproto.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_secproto_fromtext(dns_secproto_t *secprotop, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a DNSSEC security protocol value.
 * The text may contain either a mnemonic protocol name or a decimal protocol
 * number.
 *
 * Requires:
 *\li	'secprotop' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	ISC_R_SUCCESS			on success
 *\li	ISC_R_RANGE			numeric type is out of range
 *\li	DNS_R_UNKNOWN			mnemonic type is unknown
 */

isc_result_t
dns_secproto_totext(dns_secproto_t secproto, isc_buffer_t *target);
/*%<
 * Put a textual representation of the DNSSEC security protocol 'secproto'
 * into 'target'.
 *
 * Requires:
 *\li	'secproto' is a valid secproto.
 *
 *\li	'target' is a valid text buffer.
 *
 * Ensures,
 *	if the result is success:
 *	\li	The used space in 'target' is updated.
 *
 * Returns:
 *\li	ISC_R_SUCCESS			on success
 *\li	ISC_R_NOSPACE			target buffer is too small
 */

ISC_LANG_ENDDECLS

#endif /* DNS_SECPROTO_H */
