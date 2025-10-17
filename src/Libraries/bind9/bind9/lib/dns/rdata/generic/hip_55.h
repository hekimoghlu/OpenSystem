/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
/* $Id: hip_55.h,v 1.2 2009/02/26 06:09:19 marka Exp $ */

#ifndef GENERIC_HIP_5_H
#define GENERIC_HIP_5_H 1

/* RFC 5205 */

typedef struct dns_rdata_hip {
	dns_rdatacommon_t	common;
	isc_mem_t *		mctx;
	unsigned char *		hit;
	unsigned char *		key;
	unsigned char *		servers;
	isc_uint8_t		algorithm;
	isc_uint8_t		hit_len;
	isc_uint16_t		key_len;
	isc_uint16_t		servers_len;
	/* Private */
	isc_uint16_t		offset;
} dns_rdata_hip_t;

isc_result_t
dns_rdata_hip_first(dns_rdata_hip_t *);

isc_result_t
dns_rdata_hip_next(dns_rdata_hip_t *);

void
dns_rdata_hip_current(dns_rdata_hip_t *, dns_name_t *);

#endif /* GENERIC_HIP_5_H */
