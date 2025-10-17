/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#include <sys/sysctl.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("security"),
	T_META_OWNER("chrisjd")
	);

T_DECL(stack_chk_tests, "invoke the kernel stack check tests",
    T_META_ASROOT(true),
    T_META_REQUIRES_SYSCTL_EQ("kern.development", 1),
    T_META_REQUIRES_SYSCTL_NE("kern.kasan.available", 1))
{
	int ret, dummy = 1;
	ret = sysctlbyname("kern.run_stack_chk_tests", NULL, NULL, &dummy, sizeof(dummy));

	if (ret == -1 && errno == ENOENT) {
		/* sysctl not present, so skip. */
		T_PASS("kern.run_stack_chk_tests not on this platform/configuration");
	} else {
		T_ASSERT_POSIX_SUCCESS(ret, "run stack check tests");
	}
}
