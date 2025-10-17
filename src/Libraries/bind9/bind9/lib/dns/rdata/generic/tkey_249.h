/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#ifndef GENERIC_TKEY_249_H
#define GENERIC_TKEY_249_H 1

/* $Id: tkey_249.h,v 1.24 2007/06/19 23:47:17 tbox Exp $ */

/*!
 *  \brief Per draft-ietf-dnsind-tkey-00.txt */

typedef struct dns_rdata_tkey {
        dns_rdatacommon_t	common;
        isc_mem_t *		mctx;
        dns_name_t		algorithm;
        isc_uint32_t		inception;
        isc_uint32_t		expire;
        isc_uint16_t		mode;
        isc_uint16_t		error;
        isc_uint16_t		keylen;
        unsigned char *		key;
        isc_uint16_t		otherlen;
        unsigned char *		other;
} dns_rdata_tkey_t;


#endif /* GENERIC_TKEY_249_H */
