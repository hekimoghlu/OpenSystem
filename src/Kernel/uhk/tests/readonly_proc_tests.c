/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
	T_META_NAMESPACE("xnu.bsd"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("bsd"),
	T_META_OWNER("chrisjd")
	);

T_DECL(readonly_proc_tests, "invoke the read-only proc unit test",
    T_META_ASROOT(true), T_META_REQUIRES_SYSCTL_EQ("kern.development", 1))
{
	int64_t result = 0;
	int64_t value = 0;
	size_t s = sizeof(value);
	int ret;
	ret = sysctlbyname("debug.test.readonly_proc_test", &result, &s, &value, sizeof(value));
	T_ASSERT_POSIX_SUCCESS(ret, "sysctlbyname(\"debug.test.readonly_proc_test\"");
	T_EXPECT_EQ(1ull, result, "run readonly proc test");
}
