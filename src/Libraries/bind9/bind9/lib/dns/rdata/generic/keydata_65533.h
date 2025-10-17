/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#ifndef GENERIC_KEYDATA_65533_H
#define GENERIC_KEYDATA_65533_H 1

/* $Id: keydata_65533.h,v 1.2 2009/06/30 02:52:32 each Exp $ */

typedef struct dns_rdata_keydata {
	dns_rdatacommon_t	common;
	isc_mem_t *		mctx;
	isc_uint32_t		refresh;      /* Timer for refreshing data */
	isc_uint32_t		addhd;	      /* Hold-down timer for adding */
	isc_uint32_t		removehd;     /* Hold-down timer for removing */
	isc_uint16_t		flags;	      /* Copy of DNSKEY_48 */
	isc_uint8_t		protocol;
	isc_uint8_t		algorithm;
	isc_uint16_t		datalen;
	unsigned char *		data;
} dns_rdata_keydata_t;

#endif /* GENERIC_KEYDATA_65533_H */
