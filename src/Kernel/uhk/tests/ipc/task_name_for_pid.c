/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include <unistd.h>
#include <darwintest.h>
#include <mach/mach.h>
#include <drop_priv.h>
T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_TAG_VM_PREFERRED);

T_DECL(task_name_for_pid_entitlement, "Test that task_name_for_pid suceeds with entitlement",
    T_META_ASROOT(false),
    T_META_CHECK_LEAKS(false))
{
	kern_return_t kr;
	mach_port_t tname;
	pid_t pid;
	T_SETUPBEGIN;
	T_ASSERT_NE(getuid(), 0, "test should not be root uid");
	T_SETUPEND;
	// launchd has root uid/gid so we know that we must be hitting the entitlement check here.
	kr = task_name_for_pid(mach_task_self(), 1, &tname);
	T_ASSERT_MACH_SUCCESS(kr, "task_name_for_pid should succeed on launchd (pid 1)");
	pid_for_task(tname, &pid);
	T_ASSERT_EQ(pid, 1, "pid_for_task should return pid for launchd (pid 1)");

	mach_port_deallocate(mach_task_self(), tname);
}
