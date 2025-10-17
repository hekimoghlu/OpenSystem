/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#ifndef GENERIC_TXT_16_H
#define GENERIC_TXT_16_H 1

/* $Id: txt_16.h,v 1.28 2007/06/19 23:47:17 tbox Exp $ */

typedef struct dns_rdata_txt_string {
                isc_uint8_t    length;
                unsigned char   *data;
} dns_rdata_txt_string_t;

typedef struct dns_rdata_txt {
        dns_rdatacommon_t       common;
        isc_mem_t               *mctx;
        unsigned char           *txt;
        isc_uint16_t            txt_len;
        /* private */
        isc_uint16_t            offset;
} dns_rdata_txt_t;

/*
 * ISC_LANG_BEGINDECLS and ISC_LANG_ENDDECLS are already done
 * via rdatastructpre.h and rdatastructsuf.h.
 */

isc_result_t
dns_rdata_txt_first(dns_rdata_txt_t *);

isc_result_t
dns_rdata_txt_next(dns_rdata_txt_t *);

isc_result_t
dns_rdata_txt_current(dns_rdata_txt_t *, dns_rdata_txt_string_t *);

#endif /* GENERIC_TXT_16_H */
