/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#ifndef GENERIC_RT_21_H
#define GENERIC_RT_21_H 1

/* $Id: rt_21.h,v 1.21 2007/06/19 23:47:17 tbox Exp $ */

/*!
 *  \brief Per RFC1183 */

typedef struct dns_rdata_rt {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	isc_uint16_t		preference;
	dns_name_t		host;
} dns_rdata_rt_t;

#endif /* GENERIC_RT_21_H */
