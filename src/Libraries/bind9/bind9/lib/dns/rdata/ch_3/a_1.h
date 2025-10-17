/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
/* $Id: a_1.h,v 1.5 2007/06/19 23:47:17 tbox Exp $ */

/* by Bjorn.Victor@it.uu.se, 2005-05-07 */
/* Based on generic/mx_15.h */

#ifndef CH_3_A_1_H
#define CH_3_A_1_H 1

typedef isc_uint16_t ch_addr_t;

typedef struct dns_rdata_ch_a {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
  	dns_name_t		ch_addr_dom; /* ch-addr domain for back mapping */
	ch_addr_t		ch_addr; /* chaos address (16 bit) network order */
} dns_rdata_ch_a_t;

#endif /* CH_3_A_1_H */
