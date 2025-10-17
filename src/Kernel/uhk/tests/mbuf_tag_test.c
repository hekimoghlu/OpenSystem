/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_CHECK_LEAKS(false));


T_DECL(mbuf_tag_test, "test mbuf packet tags ", T_META_TAG_VM_PREFERRED)
{
	size_t len;
	int val;

	if (sysctlbyname("kern.ipc.mb_tag_test", NULL, &len, NULL, 0) != 0) {
		T_SKIP("sysctl variable kern.ipc.mb_tag_test does not exist");
		return;
	}
	val = 1;
	T_ASSERT_POSIX_SUCCESS(
		sysctlbyname("kern.ipc.mb_tag_test", NULL, NULL, &val, sizeof(val)),
		"sysctlbyname(kern.ipc.mb_tag_test)");
}
