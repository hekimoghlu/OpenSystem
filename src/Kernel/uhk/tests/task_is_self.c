/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/mach_init.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RUN_CONCURRENTLY(TRUE),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_TAG_VM_PREFERRED);

T_DECL(mach_task_is_self,
    "test task port comparison check")
{
	mach_port_t self_insp, self_read, self_name, port;

	T_ASSERT_MACH_SUCCESS(task_get_special_port(mach_task_self(), TASK_READ_PORT, &self_read), "task_get_special_port failed");
	T_ASSERT_MACH_SUCCESS(task_get_special_port(mach_task_self(), TASK_INSPECT_PORT, &self_insp), "task_get_special_port failed");
	T_ASSERT_MACH_SUCCESS(task_get_special_port(mach_task_self(), TASK_NAME_PORT, &self_name), "task_get_special_port failed");

	T_ASSERT_MACH_SUCCESS(mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port), "mach_port_allocate failed");

	T_EXPECT_NE(self_read, self_insp, "read and inspect port should be different");
	T_EXPECT_NE(self_read, mach_task_self(), "read and control port should be different");

	T_EXPECT_EQ(1, mach_task_is_self(mach_task_self()), "control port should point to self");
	T_EXPECT_EQ(1, mach_task_is_self(self_read), "read port should point to self");
	T_EXPECT_EQ(1, mach_task_is_self(self_insp), "inspect port should point to self");
	T_EXPECT_EQ(1, mach_task_is_self(self_name), "name port should point to self");
	T_EXPECT_NE(1, mach_task_is_self(port), "_port_ should not point to self");
}
