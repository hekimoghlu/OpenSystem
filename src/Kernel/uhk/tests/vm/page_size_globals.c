/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#include <mach/vm_page_size.h>

T_GLOBAL_META(
	T_META_NAMESPACE("vm_page_size_overrides")
	);

static void
verify_page_size(
	int expected_shift,
	int page_shift,
	vm_size_t page_size,
	vm_size_t page_mask)
{
	T_ASSERT_EQ(page_shift, expected_shift, "page_shift");
	T_ASSERT_EQ(page_size, 1UL << expected_shift, "page_size");
	T_ASSERT_EQ(page_mask, page_size - 1, "page_mask");
}


T_DECL(kernel_4k,
    "Can override vm_kernel_page_size",
    T_META_ENVVAR("VM_KERNEL_PAGE_SIZE_4K=1"),
    T_META_ENVVAR("MallocGuardEdges=0"),
    T_META_ENVVAR("MallocDoNotProtectPrelude=1"),
    T_META_ENVVAR("MallocDoNotProtectPostlude=1"))
{
	verify_page_size(12, vm_kernel_page_shift, vm_kernel_page_size, vm_kernel_page_mask);
}

T_DECL(invalid,
    "Invalid overrides",
    T_META_ENVVAR("VM_KERNEL_PAGE_SIZE_4K=2"),
    T_META_ENVVAR("VM_KERNEL_PAGE_SIZE=4K"),
    T_META_ENVVAR("VM_KERNEL_PAGE_SIZE="))
{
	/*
	 * This test just verifies that libkernel_init doesn't
	 * crash when handling invalid overrides.
	 * So if we got here, we can pass the test.
	 */
	T_PASS("Test process spawned");
}
