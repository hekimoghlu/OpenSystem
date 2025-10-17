/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include <sys/sysctl.h>
#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("zalloc"));

static int64_t
run_sysctl_test(const char *t, int64_t value)
{
	char name[1024];
	int64_t result = 0;
	size_t s = sizeof(value);
	int rc;

	snprintf(name, sizeof(name), "debug.test.%s", t);
	rc = sysctlbyname(name, &result, &s, &value, s);
	T_ASSERT_POSIX_SUCCESS(rc, "sysctlbyname(%s)", t);
	return result;
}

T_DECL(kalloc_type, "kalloc_type_test",
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
	T_EXPECT_EQ(1ll, run_sysctl_test("kalloc_type", 260), "test succeeded");
}

T_DECL(kalloc, "kalloc_test",
    T_META_NAMESPACE("xnu.vm"),
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
	T_EXPECT_EQ(1ll, run_sysctl_test("kalloc", 0), "test succeeded");
}

T_DECL(kalloc_guard_regions, "Checks that guard regions are inserted frequently enough",
    T_META_NAMESPACE("xnu.vm"),
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
	T_EXPECT_EQ(1ll, run_sysctl_test("kalloc_guard_regions", 0), "kalloc_guard_regions");
}
