/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#include <System/kern/remote_time.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <sys/sysctl.h>
#include <TargetConditionals.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

extern uint64_t __mach_bridge_remote_time(uint64_t);

T_DECL(remote_time_syscall, "test mach_bridge_remote_time syscall",
    T_META_CHECK_LEAKS(false))
{
#if TARGET_OS_BRIDGE
	uint64_t local_time = mach_absolute_time();
	uint64_t remote_time1 = mach_bridge_remote_time(local_time);
	uint64_t remote_time2 = __mach_bridge_remote_time(local_time);
	T_LOG("local_time = %llu, remote_time1 = %llu, remote_time2 = %llu",
	    local_time, remote_time1, remote_time2);
	T_ASSERT_EQ(remote_time1, remote_time2, "syscall works");
#else
	T_SKIP("Skipping test");
#endif /* TARGET_OS_BRIDGE */
}
