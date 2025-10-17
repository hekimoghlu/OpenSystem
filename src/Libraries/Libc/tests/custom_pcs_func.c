/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include <darwintest.h>
#include <execinfo.h>
#include <libc_private.h>

#define MAGIC ((void *)0xdeadbeef)

static void custom_thread_stack_pcs(vm_address_t *buffer, unsigned max,
               unsigned *nb, __unused unsigned skip, __unused void *startfp) {
	T_EXPECT_GE(max, 1, "need to be allowed to write at least one address for this test to be sane");

	buffer[0] = (vm_address_t)MAGIC;
	*nb = 1;
}

T_DECL(custom_pcs_func, "make sure backtrace respects custom get pcs functions")
{
	backtrace_set_pcs_func(custom_thread_stack_pcs);

	void *array[2] = { NULL, NULL };
	int nframes = backtrace(array, 2);
	T_EXPECT_EQ(nframes, 1, "custom_thread_stack_pcs should only find one pc");

	T_EXPECT_EQ(array[1], NULL, "the second pc should not be written");
	T_EXPECT_EQ(array[0], MAGIC, "the first pc magic should be %p", MAGIC);
}
