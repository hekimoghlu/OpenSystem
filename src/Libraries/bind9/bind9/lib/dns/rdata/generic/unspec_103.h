/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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
/* */
#ifndef GENERIC_UNSPEC_103_H
#define GENERIC_UNSPEC_103_H 1

/* $Id: unspec_103.h,v 1.17 2007/06/19 23:47:17 tbox Exp $ */

typedef struct dns_rdata_unspec_t {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	unsigned char		*data;
	isc_uint16_t		datalen;
} dns_rdata_unspec_t;

#endif /* GENERIC_UNSPEC_103_H */
