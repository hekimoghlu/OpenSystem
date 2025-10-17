/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
// Copyright 2021 (c) Apple Inc.  All rights reserved.

#include <darwintest.h>
#include <darwintest_posix.h>
#include <libproc.h>
#include <stdint.h>
#include <sys/resource.h>
#include <unistd.h>

#include "test_utils.h"
#include "recount_test_utils.h"

T_GLOBAL_META(
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("RM"),
    T_META_OWNER("mwidmann"),
    T_META_CHECK_LEAKS(false));

T_DECL(rusage_kernel_cpu_time_sanity,
    "ensure the CPU time for kernel_task is sane", T_META_ASROOT(true), T_META_TAG_VM_PREFERRED)
{
	struct rusage_info_v5 usage_info = { 0 };
	T_SETUPBEGIN;
	int ret = proc_pid_rusage(0, RUSAGE_INFO_V5, (void *)&usage_info);
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage on kernel_task");
	T_SETUPEND;

	T_EXPECT_GT(usage_info.ri_system_time + usage_info.ri_user_time,
	    UINT64_C(0), "kernel CPU time should be non-zero");
	if (has_user_system_times()) {
		T_EXPECT_EQ(usage_info.ri_user_time,
		    UINT64_C(0), "kernel user CPU time should be zero");
	}
}

T_DECL(rusage_user_time_sanity,
    "ensure the user CPU time for a user space task is sane", T_META_TAG_VM_PREFERRED)
{
	struct rusage_info_v5 usage_info = { 0 };
	T_SETUPBEGIN;
	int ret = proc_pid_rusage(getpid(), RUSAGE_INFO_V5, (void *)&usage_info);
	T_ASSERT_POSIX_SUCCESS(ret, "proc_pid_rusage on self");
	T_SETUPEND;

	T_EXPECT_NE(usage_info.ri_user_time, UINT64_C(0),
	    "user space CPU time should be non-zero");
	if (has_user_system_times()) {
		T_EXPECT_GT(usage_info.ri_system_time, UINT64_C(0),
		    "system time should be non-zero");
	}
}
