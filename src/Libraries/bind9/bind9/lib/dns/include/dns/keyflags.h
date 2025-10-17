/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
/* $Id: keyflags.h,v 1.16 2007/06/19 23:47:16 tbox Exp $ */

#ifndef DNS_KEYFLAGS_H
#define DNS_KEYFLAGS_H 1

/*! \file dns/keyflags.h */

#include <isc/lang.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_keyflags_fromtext(dns_keyflags_t *flagsp, isc_textregion_t *source);
/*%<
 * Convert the text 'source' refers to into a DNSSEC KEY flags value.
 * The text may contain either a set of flag mnemonics separated by
 * vertical bars or a decimal flags value.  For compatibility with
 * older versions of BIND and the DNSSEC signer, octal values
 * prefixed with a zero and hexadecimal values prefixed with "0x"
 * are also accepted.
 *
 * Requires:
 *\li	'flagsp' is a valid pointer.
 *
 *\li	'source' is a valid text region.
 *
 * Returns:
 *\li	ISC_R_SUCCESS			on success
 *\li	ISC_R_RANGE			numeric flag value is out of range
 *\li	DNS_R_UNKNOWN			mnemonic flag is unknown
 */

ISC_LANG_ENDDECLS

#endif /* DNS_KEYFLAGS_H */
