/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#include <cstdint>
#include <cstdlib>

#include "tmo_test_defs.h"

extern "C" {

struct test_struct {
	uint64_t a[64];
};

void **
cpp_new_fallback(void)
{
	void **ptrs = (void **)calloc(N_UNIQUE_CALLSITES, sizeof(void *));
	if (!ptrs) {
		return NULL;
	}

	int i = 0;
	CALL_N_CALLSITES(({ ptrs[i] = (void *)(new test_struct()); i++; }));

	return ptrs;
}

void
cpp_delete_fallback(void **ptrs)
{
	for (int i = 0; i < N_UNIQUE_CALLSITES; i++) {
		delete (test_struct *)ptrs[i];
	}

	free(ptrs);
}

}
