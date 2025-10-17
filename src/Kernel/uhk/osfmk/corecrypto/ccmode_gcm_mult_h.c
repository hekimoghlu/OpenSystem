/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#include "cc_runtime_config.h"
#include "ccaes_vng_gcm.h"
#include "ccmode_internal.h"

/*!
 *  GCM multiply by H
 *  @param key   The GCM state which holds the H value
 *  @param I     The value to multiply H by
 */
void
ccmode_gcm_mult_h(ccgcm_ctx *key, unsigned char *I)
{
#if CCMODE_GCM_VNG_SPEEDUP
#ifdef  __x86_64__
	if (!(CC_HAS_AESNI() && CC_HAS_SupplementalSSE3())) {
		//It can handle in and out buffers to be the same
		ccmode_gcm_gf_mult(CCMODE_GCM_KEY_H(key), I, I);
		return;
	} else
#endif
	{
		// CCMODE_GCM_VNG_KEY_Htable must be the second argument. gcm_gmult() is not a general multiplier function.
		gcm_gmult(I, CCMODE_GCM_VNG_KEY_Htable(key), I );
		return;
	}
#else
	//It can handle in and out buffers to be the same
	ccmode_gcm_gf_mult(CCMODE_GCM_KEY_H(key), I, I);
#endif
}

