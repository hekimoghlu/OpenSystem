/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include "cc_internal.h"
#include <corecrypto/cchmac.h>

#include "fipspost_trace.h"

void
cchmac(const struct ccdigest_info *di,
    size_t key_len, const void *key,
    size_t data_len, const void *data, unsigned char *mac)
{
	CC_ENSURE_DIT_ENABLED

	    FIPSPOST_TRACE_EVENT;

	cchmac_di_decl(di, hc);
	cchmac_init(di, hc, key_len, key);
	cchmac_update(di, hc, data_len, data);
	cchmac_final(di, hc, mac);
	cchmac_di_clear(di, hc);
}

