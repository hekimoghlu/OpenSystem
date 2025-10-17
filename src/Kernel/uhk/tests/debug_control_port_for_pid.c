/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_types.h>
#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(debug_control_port_for_pid_success,
    "Verify that with debug_port entitlement you can call debug_control_port_for_pid",
    T_META_ASROOT(true), T_META_CHECK_LEAKS(false))
{
	if (geteuid() != 0) {
		T_SKIP("test requires root privileges to run.");
	}

	mach_port_t port = MACH_PORT_NULL;
	T_ASSERT_MACH_SUCCESS(debug_control_port_for_pid(mach_task_self(), 1, &port), "debug_control_port_for_pid");
	T_EXPECT_NE(port, MACH_PORT_NULL, "debug_port");
	mach_port_deallocate(mach_task_self(), port);
}
