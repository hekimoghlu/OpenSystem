/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#include <darwintest_posix.h>
#include <mach/boolean.h>
#include <mach/vm_page_size.h>
#include <stdint.h>
#include <sys/kern_memorystatus.h>
#include <sys/sysctl.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.memorystatus"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"),
	T_META_CHECK_LEAKS(false),
	T_META_RUN_CONCURRENTLY(true),
	T_META_TAG_VM_PREFERRED,
	T_META_ASROOT(false),
	T_META_ENABLED(!TARGET_OS_OSX));

T_DECL(page_shortage_threshold_update,
    "Verify that page shortage thresholds can be read/written-to")
{
	int ret;
	uint32_t threshold_mb, expected_threshold_pages, threshold_pages;
	size_t threshold_size = sizeof(threshold_size);

	ret = sysctlbyname("kern.memorystatus.critical_threshold_mb",
	    &threshold_mb, &threshold_size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "Critical threshold can be read");
	ret = sysctlbyname("kern.memorystatus.critical_threshold_mb",
	    NULL, 0, &threshold_mb, threshold_size);
	T_ASSERT_POSIX_SUCCESS(ret, "Critical threshold can be written to");
	expected_threshold_pages = (threshold_mb << 20) / vm_kernel_page_size;
	ret = sysctlbyname("kern.memorystatus.critical_threshold_pages",
	    &threshold_pages, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.critical_threshold_pages)");
	T_EXPECT_EQ(threshold_pages, expected_threshold_pages,
	    "Critical threshold is converted to pages");

	ret = sysctlbyname("kern.memorystatus.idle_threshold_mb",
	    &threshold_mb, &threshold_size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "Idle threshold can be read");
	ret = sysctlbyname("kern.memorystatus.idle_threshold_mb",
	    NULL, 0, &threshold_mb, threshold_size);
	T_ASSERT_POSIX_SUCCESS(ret, "Idle threshold can be written to");
	expected_threshold_pages = (threshold_mb << 20) / vm_kernel_page_size;
	ret = sysctlbyname("kern.memorystatus.idle_threshold_pages",
	    &threshold_pages, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.idle_threshold_pages)");
	T_EXPECT_EQ(threshold_pages, expected_threshold_pages,
	    "Idle threshold is converted to pages");

	ret = sysctlbyname("kern.memorystatus.soft_threshold_mb",
	    &threshold_mb, &threshold_size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "Soft threshold can be read");
	ret = sysctlbyname("kern.memorystatus.soft_threshold_mb",
	    NULL, 0, &threshold_mb, threshold_size);
	T_ASSERT_POSIX_SUCCESS(ret, "Soft threshold can be written to");
	expected_threshold_pages = (threshold_mb << 20) / vm_kernel_page_size;
	ret = sysctlbyname("kern.memorystatus.soft_threshold_pages",
	    &threshold_pages, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.soft_threshold_pages)");
	T_EXPECT_EQ(threshold_pages, expected_threshold_pages,
	    "Soft threshold is converted to pages");
}

static boolean_t ballast_drained;
static uint32_t prev_offset_mb;

static void
ballast_offset_teardown(void)
{
	int ret;
	ret = sysctlbyname("kern.memorystatus.ballast_offset_mb",
	    NULL, 0, &prev_offset_mb, sizeof(prev_offset_mb));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "sysctl(kern.memorystatus.ballast_offset_mb)");
	ret = sysctlbyname("kern.memorystatus.ballast_drained",
	    NULL, 0, &ballast_drained, sizeof(ballast_drained));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "sysctl(kern.memorystatus.ballast_drained)");
}

T_DECL(ballast_offset, "Verify that the ballast offset can be set and toggled")
{
	int ret;
	uint32_t threshold_mb, expected_threshold_pages, threshold_pages;
	size_t threshold_size = sizeof(threshold_mb);
	size_t ballast_size = sizeof(ballast_drained);

	ret = sysctlbyname("kern.memorystatus.ballast_offset_mb",
	    &prev_offset_mb, &threshold_size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "Ballast offset can be read");

	threshold_mb = 128;
	ret = sysctlbyname("kern.memorystatus.ballast_offset_mb",
	    NULL, 0, &threshold_mb, threshold_size);
	T_ASSERT_POSIX_SUCCESS(ret, "Ballast offset can be written to");

	expected_threshold_pages = (threshold_mb << 20) / vm_kernel_page_size;
	ret = sysctlbyname("kern.memorystatus.ballast_offset_pages",
	    &threshold_pages, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.ballast_offset_pages)");
	T_EXPECT_EQ(threshold_pages, expected_threshold_pages,
	    "Ballast offset is converted to pages");

	ret = sysctlbyname("kern.memorystatus.ballast_drained",
	    &ballast_drained, &ballast_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "sysctl(kern.memorystatus.ballast_drained)");
	T_LOG("Ballast drained: %d", ballast_drained);

	uint32_t critical_before, idle_before, soft_before;
	ret = sysctlbyname("kern.memorystatus.soft_threshold_pages",
	    &soft_before, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.soft_threshold_pages)");
	ret = sysctlbyname("kern.memorystatus.idle_threshold_pages",
	    &idle_before, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.idle_threshold_pages)");
	ret = sysctlbyname("kern.memorystatus.critical_threshold_pages",
	    &critical_before, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.critical_threshold_pages)");
	T_LOG("Pre-toggle: crit=%u idle=%u soft=%u", critical_before, idle_before, soft_before);

	T_LOG("Toggling ballast");
	boolean_t toggled_ballast_drained = !ballast_drained;
	ret = sysctlbyname("kern.memorystatus.ballast_drained",
	    NULL, 0, &toggled_ballast_drained, sizeof(prev_offset_mb));
	T_ASSERT_POSIX_SUCCESS(ret, "sysctl(kern.memorystatus.ballast_drained");
	T_ATEND(ballast_offset_teardown);

	uint32_t critical_after, idle_after, soft_after;
	ret = sysctlbyname("kern.memorystatus.soft_threshold_pages",
	    &soft_after, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.soft_threshold_pages)");
	ret = sysctlbyname("kern.memorystatus.idle_threshold_pages",
	    &idle_after, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.idle_threshold_pages)");
	ret = sysctlbyname("kern.memorystatus.critical_threshold_pages",
	    &critical_after, &threshold_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret,
	    "sysctl(kern.memorystatus.critical_threshold_pages)");
	T_LOG("Post-toggle: crit=%u idle=%u soft=%u", critical_after, idle_after, soft_after);

	if (ballast_drained) {
		T_QUIET; T_ASSERT_GT(soft_before, soft_after, "Soft threshold decreased");
		T_EXPECT_EQ(soft_before - soft_after, threshold_pages, "Soft threshold is raised by ballast offset");
		T_QUIET; T_ASSERT_GT(idle_before, idle_after, "Idle threshold decreased");
		T_EXPECT_EQ(idle_before - idle_after, threshold_pages, "Idle threshold is raised by ballast offset");
	} else {
		T_QUIET; T_ASSERT_LT(soft_before, soft_after, "Soft threshold increased");
		T_EXPECT_EQ(soft_after - soft_before, threshold_pages, "Soft threshold is raised by ballast offset");
		T_QUIET; T_ASSERT_LT(idle_before, idle_after, "Idle threshold increased");
		T_EXPECT_EQ(idle_after - idle_before, threshold_pages, "Idle threshold is raised by ballast offset");
	}

	T_EXPECT_EQ(critical_before, critical_after, "Critical threshold is unchanged");
}
