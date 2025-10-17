/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include <sys/types.h>
#include <sys/sysctl.h>

T_DECL(vm_analytics, "Report VM analytics",
    T_META_ASROOT(true),
    T_META_REQUIRES_SYSCTL_EQ("kern.development", 1),
    T_META_TAG_VM_PREFERRED)
{
	int val = 1;
	int ret = sysctlbyname("vm.analytics_report", NULL, NULL, &val, sizeof(val));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "vm.analytics_report=1");
}
