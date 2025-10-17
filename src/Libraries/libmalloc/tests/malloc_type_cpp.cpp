/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

struct test_data_struct {
	uint64_t a[64];
};

void *
cpp_new_data(void)
{
	// Exhaust in the early allocator
	for (int i = 0; i < 1000; i++) {
		test_data_struct *p = new test_data_struct();
		delete p;
	}
	return new test_data_struct();
}

void
cpp_delete_data(void *p)
{
	delete (test_data_struct *)p;
}

struct test_ptr_struct {
	void *p[64];
};

void *
cpp_new_ptr(void)
{
	// Exhaust in the early allocator
	for (int i = 0; i < 1000; i++) {
		test_ptr_struct *p = new test_ptr_struct();
		delete p;
	}
	return new test_ptr_struct();
}

void
cpp_delete_ptr(void *p)
{
	delete (test_ptr_struct *)p;
}

void **
cpp_new_test_structs(void)
{
	void **ptrs = (void **)calloc(N_TMO_TEST_STRUCTS, sizeof(void *));
	if (!ptrs) {
		return NULL;
	}

	int i = 0;
#define tmo_new_struct(type) (({ ptrs[i] = new type(); i++; }))
	FOREACH_TMO_TEST_STRUCT(INVOKE_FOR_TMO_TEST_STRUCT_CPP_TYPE,
			tmo_new_struct);

	return ptrs;
}

void
cpp_delete_test_structs(void **ptrs)
{
	int i = 0;
#define tmo_delete_struct(type) (({ delete (type *)ptrs[i]; i++; }))
	FOREACH_TMO_TEST_STRUCT(INVOKE_FOR_TMO_TEST_STRUCT_CPP_TYPE,
			tmo_delete_struct);

	free(ptrs);
}

}
