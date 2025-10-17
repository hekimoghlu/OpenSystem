/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#include "ccdrbg.h"
#include "ccdrbg_internal.h"

bool
ccdrbg_must_reseed(const struct ccdrbg_info *info,
    const struct ccdrbg_state *drbg)
{
	CC_ENSURE_DIT_ENABLED

	return info->must_reseed(drbg);
}

int
ccdrbg_init(const struct ccdrbg_info *info,
    struct ccdrbg_state *drbg,
    size_t entropyLength, const void* entropy,
    size_t nonceLength, const void* nonce,
    size_t psLength, const void* ps)
{
	CC_ENSURE_DIT_ENABLED

	return info->init(info, drbg, entropyLength, entropy, nonceLength, nonce, psLength, ps);
}

int
ccdrbg_reseed(const struct ccdrbg_info *info,
    struct ccdrbg_state *drbg,
    size_t entropyLength, const void *entropy,
    size_t additionalLength, const void *additional)
{
	CC_ENSURE_DIT_ENABLED

	return info->reseed(drbg, entropyLength, entropy, additionalLength, additional);
}


int
ccdrbg_generate(const struct ccdrbg_info *info,
    struct ccdrbg_state *drbg,
    size_t dataOutLength, void *dataOut,
    size_t additionalLength, const void *additional)
{
	CC_ENSURE_DIT_ENABLED

	return info->generate(drbg, dataOutLength, dataOut, additionalLength, additional);
}

void
ccdrbg_done(const struct ccdrbg_info *info, struct ccdrbg_state *drbg)
{
	info->done(drbg);
}

size_t
ccdrbg_context_size(const struct ccdrbg_info *info)
{
	return info->size;
}

