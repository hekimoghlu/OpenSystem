/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#ifndef IN_1_SRV_33_H
#define IN_1_SRV_33_H 1

/* $Id: srv_33.h,v 1.19 2007/06/19 23:47:17 tbox Exp $ */

/* Reviewed: Fri Mar 17 13:01:00 PST 2000 by bwelling */

/*! 
 *  \brief Per RFC2782 */

typedef struct dns_rdata_in_srv {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	isc_uint16_t		priority;
	isc_uint16_t		weight;
	isc_uint16_t		port;
	dns_name_t		target;
} dns_rdata_in_srv_t;

#endif /* IN_1_SRV_33_H */
