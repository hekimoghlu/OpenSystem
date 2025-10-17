/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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
#include "ccsha2_internal.h"
#include "cc_runtime_config.h"
#include "fipspost_trace.h"

const struct ccdigest_info *
ccsha256_di(void)
{
	FIPSPOST_TRACE_EVENT;

#if CC_USE_ASM && CCSHA2_VNG_INTEL
#if defined(__x86_64__)
	if (CC_HAS_AVX512_AND_IN_KERNEL()) {
		return &ccsha256_vng_intel_SupplementalSSE3_di;
	}

	if (CC_HAS_AVX2()) {
		return &ccsha256_vng_intel_AVX2_di;
	}

	if (CC_HAS_AVX1()) {
		return &ccsha256_vng_intel_AVX1_di;
	}
#endif

	return &ccsha256_vng_intel_SupplementalSSE3_di;
#elif CC_USE_ASM && CCSHA2_VNG_ARM
	return &ccsha256_vng_arm_di;
#elif CCSHA256_ARMV6M_ASM
	return &ccsha256_v6m_di;
#else
	return &ccsha256_ltc_di;
#endif
}

