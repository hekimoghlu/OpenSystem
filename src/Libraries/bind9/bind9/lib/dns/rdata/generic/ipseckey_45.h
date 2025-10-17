/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
/* $Id: ipseckey_45.h,v 1.4 2007/06/19 23:47:17 tbox Exp $ */

#ifndef GENERIC_IPSECKEY_45_H
#define GENERIC_IPSECKEY_45_H 1

typedef struct dns_rdata_ipseckey {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	isc_uint8_t		precedence;
	isc_uint8_t		gateway_type;
	isc_uint8_t		algorithm;
	struct in_addr		in_addr;	/* gateway type 1 */
	struct in6_addr		in6_addr;	/* gateway type 2 */
	dns_name_t		gateway;	/* gateway type 3 */
	unsigned char		*key;
	isc_uint16_t		keylength;
} dns_rdata_ipseckey_t;

#endif /* GENERIC_IPSECKEY_45_H */
