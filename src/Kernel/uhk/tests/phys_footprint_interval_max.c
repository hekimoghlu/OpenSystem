/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <sys/resource.h>
#include <libproc.h>
#include <libproc_internal.h>
#include <TargetConditionals.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

#define ALLOC_SIZE_LARGE 5*1024*1024
#define ALLOC_SIZE_SMALL 2*1024*1024

int proc_rlimit_control(pid_t pid, int flavor, void *arg);

T_DECL(phys_footprint_interval_max,
    "Validate physical footprint interval tracking", T_META_TAG_VM_PREFERRED)
{
	int ret;
	struct rusage_info_v4 ru;
	mach_vm_address_t addr = (mach_vm_address_t)NULL;

	ret = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ru);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage");
	T_ASSERT_EQ(ru.ri_lifetime_max_phys_footprint, ru.ri_interval_max_phys_footprint,
	    "Max footprint and interval footprint are equal prior to dirtying memory");

	ret = mach_vm_allocate(mach_task_self(), &addr, (mach_vm_size_t)ALLOC_SIZE_LARGE, VM_FLAGS_ANYWHERE);
	T_QUIET;
	T_ASSERT_MACH_SUCCESS(ret, "mach_vm_allocate(ALLOC_SIZE_LARGE)");

	memset((void *)addr, 0xab, ALLOC_SIZE_LARGE);

	ret = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ru);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage");
	T_ASSERT_EQ(ru.ri_lifetime_max_phys_footprint, ru.ri_interval_max_phys_footprint,
	    "Max footprint and interval footprint are equal after dirtying large memory region");

	mach_vm_deallocate(mach_task_self(), addr, (mach_vm_size_t)ALLOC_SIZE_LARGE);

	ret = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ru);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage");
	T_ASSERT_EQ(ru.ri_lifetime_max_phys_footprint, ru.ri_interval_max_phys_footprint,
	    "Max footprint and interval footprint are still equal after freeing large memory region");

	ret = proc_reset_footprint_interval(getpid());
	T_ASSERT_POSIX_SUCCESS(ret, "proc_reset_footprint_interval()");

	ret = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ru);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage");
	T_ASSERT_GT(ru.ri_lifetime_max_phys_footprint, ru.ri_interval_max_phys_footprint,
	    "Max footprint is greater than interval footprint after resetting interval");

	ret = mach_vm_allocate(mach_task_self(), &addr, (mach_vm_size_t)ALLOC_SIZE_SMALL, VM_FLAGS_ANYWHERE);
	T_QUIET;
	T_ASSERT_MACH_SUCCESS(ret, "mach_vm_allocate(ALLOC_SIZE_SMALL)");
	memset((void *)addr, 0xab, ALLOC_SIZE_SMALL);

	ret = proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ru);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage");
	T_ASSERT_GT(ru.ri_lifetime_max_phys_footprint, ru.ri_interval_max_phys_footprint,
	    "Max footprint is still greater than interval footprint after dirtying small memory region");
}
