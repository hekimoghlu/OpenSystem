/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#include <sys/types.h>
#include <sys/sysctl.h>
#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_ASROOT(false)
	);

static void
test_sysctl(const char* name)
{
	int value, previous_value = 0, current_value = 0;
	size_t len = sizeof(value);

	T_ASSERT_POSIX_SUCCESS(sysctlbyname(name, &value, &len, NULL, 0), "Get current value of sysctl %s", name);
	previous_value = value;
	value = 66;
	T_ASSERT_POSIX_SUCCESS(sysctlbyname(name, NULL, NULL, &value, sizeof(value)), "Set value of sysctl %s, prev=%d new=%d", name, previous_value, value);
	T_ASSERT_POSIX_SUCCESS(sysctlbyname(name, &current_value, &len, NULL, 0), "Get new value of sysctl %s", name);
	T_ASSERT_EQ(value, current_value, "Verify value was actually set");
	T_ASSERT_POSIX_SUCCESS(sysctlbyname(name, NULL, NULL, &previous_value, sizeof(previous_value)), "Restore value of sysctl %s to %d", name, previous_value);
}

T_DECL(nework_elevated_logging, "Tests enforcement of entitlement as non-root")
{
	test_sysctl("net.route.verbose");
	test_sysctl("net.inet6.icmp6.nd6_debug");
}
