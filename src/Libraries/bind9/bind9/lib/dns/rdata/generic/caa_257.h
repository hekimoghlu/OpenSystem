/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#ifndef GENERIC_CAA_257_H
#define GENERIC_CAA_257_H 1

/* $Id$ */

typedef struct dns_rdata_caa {
	dns_rdatacommon_t	common;
	isc_mem_t *		mctx;
	isc_uint8_t		flags;
	unsigned char *		tag;
	isc_uint8_t		tag_len;
	unsigned char		*value;
	isc_uint8_t		value_len;
} dns_rdata_caa_t;

#endif /* GENERIC_CAA_257_H */
