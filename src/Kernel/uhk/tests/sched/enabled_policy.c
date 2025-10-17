/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/sysctl.h>

#include <darwintest.h>
#include "test_utils.h"
#include "sched_test_utils.h"

T_GLOBAL_META(T_META_NAMESPACE("xnu.scheduler"),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("scheduler"),
    T_META_TAG_VM_NOT_ELIGIBLE);

static void
get_device_name(char *device_name, size_t device_name_len)
{
	int ret;
	ret = sysctlbyname("hw.target", device_name, &device_name_len, NULL, 0);
	T_QUIET; T_ASSERT_EQ(ret, 0, "sysctlbyname hw.target");
}

static void
get_kern_version(char *kern_version, size_t kern_version_len)
{
	int ret;
	ret = sysctlbyname("kern.version", kern_version, &kern_version_len, NULL, 0);
	T_QUIET; T_ASSERT_EQ(ret, 0, "sysctlbyname kern.version");
}

static bool
platform_is_arm64(void)
{
	int ret;
	int is_arm64 = 0;
	size_t is_arm64_size = sizeof(is_arm64);
	ret = sysctlbyname("hw.optional.arm64", &is_arm64, &is_arm64_size, NULL, 0);
	return ret == 0 && is_arm64;
}


T_DECL(enabled_policy, "Verify that the expected scheduler policy is running", XNU_T_META_SOC_SPECIFIC)
{
	char *policy_name = platform_sched_policy();

	size_t device_name_len = 256;
	char device_name[device_name_len];
	get_device_name(device_name, device_name_len);
	T_LOG("Current device: %s", device_name);

	size_t kern_version_len = 256;
	char kern_version[kern_version_len];
	get_kern_version(kern_version, kern_version_len);
	T_LOG("Kernel version: %s", kern_version);

	if (!platform_is_arm64()) {
		T_SKIP("Skipping test on non-arm64 platform");
	}
	if (strstr(device_name, "DEV") != NULL) {
		T_SKIP("Skipping test on DEV hardware");
	}
	if (strstr(device_name, "SIM") != NULL) {
		T_SKIP("Skipping test on simulator");
	}

	if (!platform_is_amp()) {
		T_ASSERT_EQ_STR(policy_name, "clutch", "SMP platform should be running the Clutch scheduler");
		T_END;
	}


	T_ASSERT_EQ_STR(policy_name, "edge", "Non-exempt AMP platform should be running the Edge scheduler");
}
