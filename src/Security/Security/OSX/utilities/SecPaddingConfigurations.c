/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
#define SecPaddingDomain CFSTR("com.apple.security.padding")

typedef enum {
	SecPaddingErrorUnknownType     = -1
} SecPaddingError;

#include "debugging.h"
#include "SecCFError.h"
#include "SecCFWrappers.h"
#include <Security/SecPaddingConfigurationsPriv.h>

#pragma mark Padding Helper Methods

// Compute the next power of two
// Requires: v <= UINT64_MAX/2
static uint64_t nextPowerOfTwo(uint64_t v)
{
	if (v > (UINT64_MAX>>1)) {
		secerror("Overflowing uint64_t by requesting nextPowerOfTwo of: %llx", v);
		assert(0);
	}
	if (v & (v - 1)) {
		// Not already a power of 2
		return ((uint64_t)1 << ((int)sizeof(v)*8 - __builtin_clzll(v)));
	} else {
		// Already a power of 2
		return v;
	}
}

// Round to a multiple of n
// Requires: v+n <= UINT64_MAX
static uint64_t nextMultiple(uint64_t v,uint64_t n)
{
	// Multiples of 0 are 0. Preventing division by 0.
	if (n == 0) {
		return 0;
	}

	if (n <= 0 || v > (UINT64_MAX-n)) {
		secerror("Overflowing uint64_t by requesting nextMutiple with parameters v: %llx and n: %llx", v, n);
		assert(0);
	}
	return n*((v+n-1)/n);
}

#pragma mark Padding Configurations

int64_t SecPaddingCompute(SecPaddingType type, uint32_t size, CFErrorRef *error) {
	if (type != SecPaddingTypeMMCS) {
		if (error) {
			*error = CFErrorCreate(CFAllocatorGetDefault(), SecPaddingDomain, SecPaddingErrorUnknownType, NULL);
		}
		return SecPaddingErrorUnknownType;
	}

	int64_t paddedSize = 0;

	if (size <= 64){
		paddedSize = 64;
	} else if (size <= 1024) {
		paddedSize = nextPowerOfTwo(size);
	} else if (size <= 32000) {
		paddedSize = nextMultiple(size, 1024);
	} else {
		paddedSize = nextMultiple(size, 8192);
	}
	
	assert(paddedSize >= size);
	return paddedSize - size;
}
