/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#ifndef GENERIC_CSYNC_62_H
#define GENERIC_CSYNC_62_H 1

/*!
 * \brief Per RFC 7477
 */

typedef struct dns_rdata_csync {
	dns_rdatacommon_t	common;
	isc_mem_t		*mctx;
	isc_uint32_t		serial;
	isc_uint16_t		flags;
	unsigned char		*typebits;
	isc_uint16_t		len;
} dns_rdata_csync_t;

#endif /* GENERIC_CSYNC_62_H */
