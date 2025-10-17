/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
/* $Id: cert_37.h,v 1.20 2007/06/19 23:47:17 tbox Exp $ */

#ifndef GENERIC_CERT_37_H
#define GENERIC_CERT_37_H 1

/*% RFC2538 */
typedef struct dns_rdata_cert {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	isc_uint16_t		type;
	isc_uint16_t		key_tag;
	isc_uint8_t		algorithm;
	isc_uint16_t		length;
	unsigned char		*certificate;
} dns_rdata_cert_t;

#endif /* GENERIC_CERT_37_H */
