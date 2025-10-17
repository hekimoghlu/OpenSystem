/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include <pthread.h>
#include <machine/cpu_capabilities.h>
#include <sys/commpage.h>
#include <sys/sysctl.h>
#include <mach/vm_param.h>
#include <stdint.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.arm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("jeffrey_crowell"),
	T_META_RUN_CONCURRENTLY(true));

T_DECL(asb_comm_page_sanity,
    "Test that asb comm page values are sane.")
{
	int rv;
	uint64_t max_userspace_address = MACH_VM_MAX_ADDRESS;
	uint64_t target_address = COMM_PAGE_READ(uint64_t, ASB_TARGET_ADDRESS);

	uint64_t sysctl_min_kernel_address = 0;
	size_t min_kernel_address_size = sizeof(sysctl_min_kernel_address);
	rv = sysctlbyname("vm.vm_min_kernel_address", &sysctl_min_kernel_address, &min_kernel_address_size, NULL, 0);
	uint64_t kern_target_address = COMM_PAGE_READ(uint64_t, ASB_TARGET_KERN_ADDRESS);

	T_QUIET; T_ASSERT_GT(target_address, max_userspace_address, "check that asb target addresses are as expected");
	T_QUIET; T_ASSERT_LT(kern_target_address, sysctl_min_kernel_address, "check that asb target kernel addresses are as expected");
}
