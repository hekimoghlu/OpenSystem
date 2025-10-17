/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
/* $Id: tsig_250.h,v 1.25 2007/06/19 23:47:17 tbox Exp $ */

#ifndef ANY_255_TSIG_250_H
#define ANY_255_TSIG_250_H 1

/*% RFC2845 */
typedef struct dns_rdata_any_tsig {
	dns_rdatacommon_t	common;
	isc_mem_t *		mctx;
	dns_name_t		algorithm;
	isc_uint64_t		timesigned;
	isc_uint16_t		fudge;
	isc_uint16_t		siglen;
	unsigned char *		signature;
	isc_uint16_t		originalid;
	isc_uint16_t		error;
	isc_uint16_t		otherlen;
	unsigned char *		other;
} dns_rdata_any_tsig_t;

#endif /* ANY_255_TSIG_250_H */
