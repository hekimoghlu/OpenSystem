/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#include <TargetConditionals.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <mach/mach_error.h>
#include <mach/mach_init.h>
#include <mach/mach_vm.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"));

T_DECL(wire_copy_share,
    "test VM object wired, copied and shared", T_META_TAG_VM_PREFERRED)
{
	kern_return_t kr;
	mach_vm_address_t vmaddr1, vmaddr2, vmaddr3;
	mach_vm_size_t vmsize;
	char *cp;
	int i;
	vm_prot_t cur_prot, max_prot;
	int ret;

	/* allocate anonymous memory */
	vmaddr1 = 0;
	vmsize = 32 * PAGE_SIZE;
	kr = mach_vm_allocate(
		mach_task_self(),
		&vmaddr1,
		vmsize,
		VM_FLAGS_ANYWHERE);
	T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_allocate()");

	/* populate it */
	cp = (char *)(uintptr_t)vmaddr1;
	for (i = 0; i < vmsize; i += PAGE_SIZE) {
		cp[i] = i;
	}

	/* wire one page */
	ret = mlock(cp, PAGE_SIZE);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "mlock()");

	/* create a range to receive a copy */
	vmaddr2 = 0;
	kr = mach_vm_allocate(
		mach_task_self(),
		&vmaddr2,
		vmsize - PAGE_SIZE,
		VM_FLAGS_ANYWHERE);
	T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_allocate() for copy");

	/* copy the rest of the original object */
	kr = mach_vm_copy(
		mach_task_self(),
		vmaddr1 + PAGE_SIZE,
		vmsize - PAGE_SIZE,
		vmaddr2);
	T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_copy()");

	/* share the whole thing */
	vmaddr3 = 0;
	kr = mach_vm_remap(
		mach_task_self(),
		&vmaddr3,
		vmsize,
		0, /* mask */
		VM_FLAGS_ANYWHERE,
		mach_task_self(),
		vmaddr1,
		FALSE, /* copy */
		&cur_prot,
		&max_prot,
		VM_INHERIT_DEFAULT);
	T_QUIET; T_ASSERT_MACH_SUCCESS(kr, "vm_remap()");
}
