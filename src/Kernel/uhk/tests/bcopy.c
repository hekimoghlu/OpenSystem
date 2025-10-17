/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

T_DECL(validate_memmove_semantics,
    "Ensure xnu's platform-specific memmove() implementation follows the correct semantics",
    T_META_NAMESPACE("xnu.bcopy"),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("all"),
    T_META_TAG_VM_PREFERRED
    ) {
	// When we invoke our sysctl that exercises the in-kernel memmove implementation
	int ret = sysctlbyname("debug.test.test_memmove", NULL, NULL, NULL, 0);
	// Then the kernel reports that everything looks good
	T_ASSERT_POSIX_SUCCESS(ret, "test_memmove sysctl");
}
