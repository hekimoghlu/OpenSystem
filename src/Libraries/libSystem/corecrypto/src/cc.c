/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#include <corecrypto/cc.h>
#include <corecrypto/ccrng_system.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void cc_clear(size_t len, void *dst)
{
	volatile unsigned char *ptr = (volatile unsigned char*) dst;
	while (len--) {
		*ptr++ = 0;
	}
}

/* https://cryptocoding.net/index.php/Coding_rules#Avoid_branchings_controlled_by_secret_data */
void* cc_muxp(int s, const void *a, const void *b)
{
	uintptr_t mask = -(s != 0);
	uintptr_t ret = mask & (((uintptr_t)a) ^ ((uintptr_t)b));
	ret = ret ^ ((uintptr_t)b);
	return (void*) ret;
}

/* https://cryptocoding.net/index.php/Coding_rules#Compare_secret_strings_in_constant_time */
int cc_cmp_safe(size_t size, const void* a, const void* b)
{
	if ( size == 0) {
		return 1;
	}
	const unsigned char *_a = (const unsigned char *) a;
	const unsigned char *_b = (const unsigned char *) b;
	unsigned char result = 0;
	size_t i;

	for (i = 0; i < size; i++)
	{
		result |=  (unsigned) _a[i] ^ _b[i];
	}

	return result ? 1 : 0;
}


