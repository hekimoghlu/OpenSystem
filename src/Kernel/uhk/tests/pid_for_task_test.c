/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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

T_DECL(pid_for_task_test, "Test pid_for_task with task name port")
{
	kern_return_t kr;
	mach_port_t tname;
	pid_t pid;

	kr = task_name_for_pid(mach_task_self(), getpid(), &tname);
	T_EXPECT_EQ(kr, 0, "task_name_for_pid should succeed on current pid");
	pid_for_task(tname, &pid);
	T_EXPECT_EQ(pid, getpid(), "pid_for_task should return the same value as getpid()");

	mach_port_deallocate(mach_task_self(), tname);
}
