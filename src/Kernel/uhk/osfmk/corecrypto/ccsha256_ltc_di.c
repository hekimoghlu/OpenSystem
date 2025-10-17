/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#include <corecrypto/ccsha2.h>
#include <corecrypto/ccdigest_priv.h>
#include "ccdigest_internal.h"
#include "ccsha2_internal.h"

#if !CC_KERNEL || !CC_USE_ASM

const struct ccdigest_info ccsha256_ltc_di = {
	.output_size = CCSHA256_OUTPUT_SIZE,
	.state_size = CCSHA256_STATE_SIZE,
	.block_size = CCSHA256_BLOCK_SIZE,
	.oid_size = ccoid_sha256_len,
	.oid = CC_DIGEST_OID_SHA256,
	.initial_state = ccsha256_initial_state,
	.compress = ccsha256_ltc_compress,
	.final = ccdigest_final_64be,
	.impl = CC_IMPL_SHA256_LTC,
};

#endif

