/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#include <sys/errno.h>
#include <sys/kern_memorystatus.h>
#include <unistd.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"));

T_DECL(ios13extended_footprint_entitled, "Verify entitled memory limit can be set and queried", T_META_TAG_VM_PREFERRED)
{
	int ret;
	uint64_t memsize = 0;
	size_t memsize_size = sizeof(memsize);
	int32_t ios13extended_footprint_limit_mb = 0;
	size_t ios13extended_footprint_limit_mb_size = sizeof(ios13extended_footprint_limit_mb);

	memorystatus_memlimit_properties2_t mmprops;

	ret = sysctlbyname("hw.memsize", &memsize, &memsize_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "call sysctlbyname to get memsize.");

	if (memsize < 1500ULL * 1024 * 1024 ||
	    memsize > 2ULL * 1024 * 1024 * 1024) {
		T_SKIP("This entitlement is only supported on 2GB devices");
	}

	ret = sysctlbyname("kern.ios13extended_footprint_limit_mb", &ios13extended_footprint_limit_mb, &ios13extended_footprint_limit_mb_size, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "call sysctlbyname to get extended limit.");

	mmprops.v1.memlimit_active = -1;
	mmprops.v1.memlimit_inactive = -1;
	ret = memorystatus_control(MEMORYSTATUS_CMD_SET_MEMLIMIT_PROPERTIES, getpid(), 0, &mmprops.v1, sizeof(mmprops.v1));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "memorystatus_control");

	/* Check our memlimt */
	ret = memorystatus_control(MEMORYSTATUS_CMD_GET_MEMLIMIT_PROPERTIES, getpid(), 0, &mmprops, sizeof(mmprops));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "memorystatus_control");

	T_QUIET; T_ASSERT_EQ(mmprops.v1.memlimit_active, ios13extended_footprint_limit_mb, "active limit");
	T_QUIET; T_ASSERT_EQ(mmprops.v1.memlimit_inactive, ios13extended_footprint_limit_mb, "inactive limit");
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
 */ret = memorystatus_control(MEMORYSTATUS_CMD_CONVERT_MEMLIMIT_MB, getpid(), (uint32_t) -1, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "memorystatus_control");
	T_QUIET; T_ASSERT_EQ(ret, ios13extended_footprint_limit_mb, "got extended footprint");
}

