/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#include <pthread/private.h>
#include <sys/sysctl.h>
#include <mach/task.h>
#include "exc_helpers.h"

#define EXCEPTION_IDENTITY_PROTECTED 4

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_RUN_CONCURRENTLY(true));

static size_t
exc_immovable_handler(
	mach_port_t task,
	mach_port_t thread,
	__unused exception_type_t type,
	__unused mach_exception_data_t codes)
{
	T_EXPECT_EQ(task, mach_task_self(), "Received immovable task port");
	T_EXPECT_EQ(thread, pthread_mach_thread_np(pthread_main_thread_np()),
	    "Received immovable thread port");
	T_END;
}

static size_t
exc_handler_identity_protected(
	task_id_token_t token,
	uint64_t thread_id,
	exception_type_t type,
	__unused exception_data_t codes)
{
	mach_port_t port1, port2;
	kern_return_t kr;

	T_LOG("Got protected exception!");

	port1 = mach_task_self();
	kr = task_identity_token_get_task_port(token, TASK_FLAVOR_CONTROL, &port2); /* Immovable control port for self */
	T_ASSERT_MACH_SUCCESS(kr, "task_identity_token_get_task_port() - CONTROL");
	T_EXPECT_EQ(port1, port2, "Control port matches!");

	T_END;
}

T_DECL(exc_immovable, "Test that exceptions receive immovable ports",
    T_META_TAG_VM_PREFERRED)
{
	mach_port_t exc_port = create_exception_port(EXC_MASK_BAD_ACCESS);
	uint32_t opts = 0;
	size_t size = sizeof(&opts);
	mach_port_t mp;
	kern_return_t kr;

	T_LOG("Check if task_exc_guard exception has been enabled\n");
	int ret = sysctlbyname("kern.ipc_control_port_options", &opts, &size, NULL, 0);
	T_EXPECT_POSIX_SUCCESS(ret, "sysctlbyname(kern.ipc_control_port_options)");

	if ((opts & 0x30) == 0) {
		T_SKIP("immovable rights aren't enabled");
	}

	kr = task_get_special_port(mach_task_self(), TASK_KERNEL_PORT, &mp);
	T_EXPECT_MACH_SUCCESS(kr, "task_get_special_port");
	T_EXPECT_NE(mp, mach_task_self(), "should receive movable port");

	/*
	 * do not deallocate the port we received on purpose to check
	 * that the exception will not coalesce with the movable port
	 * we have in our space now
	 */

	run_exception_handler(exc_port, exc_immovable_handler);
	*(void *volatile*)0 = 0;
}

T_DECL(exc_raise_identity_protected, "Test identity-protected exception delivery behavior",
    T_META_TAG_VM_NOT_PREFERRED)
{
	mach_port_t exc_port = create_exception_port_behavior64(EXC_MASK_BAD_ACCESS, EXCEPTION_IDENTITY_PROTECTED);

	run_exception_handler_behavior64(exc_port, NULL, exc_handler_identity_protected, EXCEPTION_IDENTITY_PROTECTED, true);
	*(void *volatile*)0 = 0;
}
