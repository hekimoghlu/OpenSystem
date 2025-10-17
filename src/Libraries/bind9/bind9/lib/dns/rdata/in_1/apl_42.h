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
/* */
#ifndef IN_1_APL_42_H
#define IN_1_APL_42_H 1

/* $Id: apl_42.h,v 1.6 2007/06/19 23:47:17 tbox Exp $ */

typedef struct dns_rdata_apl_ent {
	isc_boolean_t	negative;
	isc_uint16_t	family;
	isc_uint8_t	prefix;
	isc_uint8_t	length;
	unsigned char	*data;
} dns_rdata_apl_ent_t;

typedef struct dns_rdata_in_apl {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	/* type & class specific elements */
	unsigned char           *apl;
        isc_uint16_t            apl_len;
        /* private */
        isc_uint16_t            offset;
} dns_rdata_in_apl_t;

/*
 * ISC_LANG_BEGINDECLS and ISC_LANG_ENDDECLS are already done
 * via rdatastructpre.h and rdatastructsuf.h.
 */

isc_result_t
dns_rdata_apl_first(dns_rdata_in_apl_t *);

isc_result_t
dns_rdata_apl_next(dns_rdata_in_apl_t *);

isc_result_t
dns_rdata_apl_current(dns_rdata_in_apl_t *, dns_rdata_apl_ent_t *);

#endif /* IN_1_APL_42_H */
