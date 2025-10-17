/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#ifndef GENERIC_LOC_29_H
#define GENERIC_LOC_29_H 1

/* $Id: loc_29.h,v 1.19 2007/06/19 23:47:17 tbox Exp $ */

/*!
 * \brief Per RFC1876 */

typedef struct dns_rdata_loc_0 {
	isc_uint8_t	version;	/* must be first and zero */
	isc_uint8_t	size;
	isc_uint8_t	horizontal;
	isc_uint8_t	vertical;
	isc_uint32_t	latitude;
	isc_uint32_t	longitude;
	isc_uint32_t	altitude;
} dns_rdata_loc_0_t;

typedef struct dns_rdata_loc {
	dns_rdatacommon_t	common;
	union {
		dns_rdata_loc_0_t v0;
	} v;
} dns_rdata_loc_t;

#endif /* GENERIC_LOC_29_H */
