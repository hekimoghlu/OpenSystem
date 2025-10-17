/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#include <corecrypto/cc_priv.h>

int
cc_cmp_safe(size_t num, const void * ptr1, const void * ptr2)
{
	CC_ENSURE_DIT_ENABLED

	size_t i;
	const uint8_t *s = (const uint8_t *)ptr1;
	const uint8_t *t = (const uint8_t *)ptr2;
	uint8_t flag = ((num <= 0)?1:0); // If 0 return an error
	for (i = 0; i < num; i++) {
		flag |= (s[i] ^ t[i]);
	}
	CC_HEAVISIDE_STEP(flag, flag); // flag=(flag==0)?0:1;
	return flag; // 0 iff all bytes were equal, 1 if there is any difference
}

