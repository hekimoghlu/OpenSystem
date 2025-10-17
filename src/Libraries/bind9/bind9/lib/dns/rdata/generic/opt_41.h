/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#ifndef GENERIC_OPT_41_H
#define GENERIC_OPT_41_H 1

/* $Id: opt_41.h,v 1.18 2007/06/19 23:47:17 tbox Exp $ */

/*!
 *  \brief Per RFC2671 */

typedef struct dns_rdata_opt_opcode {
		isc_uint16_t	opcode;
		isc_uint16_t	length;
		unsigned char	*data;
} dns_rdata_opt_opcode_t;

typedef struct dns_rdata_opt {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	unsigned char		*options;
	isc_uint16_t		length;
	/* private */
	isc_uint16_t		offset;
} dns_rdata_opt_t;

/*
 * ISC_LANG_BEGINDECLS and ISC_LANG_ENDDECLS are already done
 * via rdatastructpre.h and rdatastructsuf.h.
 */

isc_result_t
dns_rdata_opt_first(dns_rdata_opt_t *);

isc_result_t
dns_rdata_opt_next(dns_rdata_opt_t *);

isc_result_t
dns_rdata_opt_current(dns_rdata_opt_t *, dns_rdata_opt_opcode_t *);

#endif /* GENERIC_OPT_41_H */
