/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#include <stdlib.h>
#include <stdio.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true),
    T_META_NAMESPACE("xnu.ipc"),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("IPC"));

T_DECL(mach_port_mod_refs, "mach_port_mod_refs", T_META_TAG_VM_PREFERRED){
	mach_port_t port_set;
	mach_port_t port;
	task_exc_guard_behavior_t old, new;
	int ret;

	ret = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_PORT_SET, &port_set);
	T_ASSERT_MACH_SUCCESS(ret, "mach_port_allocate MACH_PORT_RIGHT_PORT_SET");

	ret = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port);
	T_ASSERT_MACH_SUCCESS(ret, "mach_port_allocate MACH_PORT_RIGHT_RECEIVE");

	/*
	 * Disable [optional] Mach port guard exceptions to avoid fatal crash
	 */
	ret = task_get_exc_guard_behavior(mach_task_self(), &old);
	T_ASSERT_MACH_SUCCESS(ret, "task_get_exc_guard_behavior");
	new = (old & ~TASK_EXC_GUARD_MP_DELIVER);
	ret = task_set_exc_guard_behavior(mach_task_self(), new);
	T_ASSERT_MACH_SUCCESS(ret, "task_set_exc_guard_behavior new");

	/*
	 * Test all known variants of port rights on each type of port
	 */

	/* can't subtract a send right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs SEND: -1 on a RECV right");

	/* can't subtract a send once right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND_ONCE, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs SEND_ONCE: -1 on a RECV right");

	/* can't subtract a PORT SET right if it's not a port set */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_PORT_SET, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs PORT_SET: -1 on a RECV right");

	/* can't subtract a dead name right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_DEAD_NAME, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs DEAD_NAME: -1 on a RECV right");

	/* can't subtract a LABELH right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_LABELH, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs LABELH: -1 on a RECV right");

	/* can't subtract an invalid right-type */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_NUMBER, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_VALUE, "mach_port_mod_refs NUMBER: -1 on a RECV right");

	/* can't subtract an invalid right-type */
	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_NUMBER + 1, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_VALUE, "mach_port_mod_refs NUMBER+1: -1 on a RECV right");


	/* can't subtract a send right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_SEND, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs SEND: -1 on a PORT_SET right");

	/* can't subtract a send once right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_SEND_ONCE, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs SEND_ONCE: -1 on a PORT_SET right");

	/* can't subtract a receive right if it's a port set */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_RECEIVE, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs RECV: -1 on a PORT_SET right");

	/* can't subtract a dead name right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_DEAD_NAME, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs DEAD_NAME: -1 on a PORT_SET right");

	/* can't subtract a LABELH right if it doesn't exist */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_LABELH, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_RIGHT, "mach_port_mod_refs LABELH: -1 on a PORT_SET right");

	/* can't subtract an invalid right-type */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_NUMBER, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_VALUE, "mach_port_mod_refs NUMBER: -1 on a PORT_SET right");

	/* can't subtract an invalid right-type */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_NUMBER + 1, -1);
	T_ASSERT_EQ(ret, KERN_INVALID_VALUE, "mach_port_mod_refs NUMBER+1: -1 on a PORT_SET right");

	/* restore the old guard behavior */
	ret = task_set_exc_guard_behavior(mach_task_self(), old);
	T_ASSERT_MACH_SUCCESS(ret, "task_set_exc_guard_behavior old");

	/*
	 * deallocate the ports/sets
	 */
	ret = mach_port_mod_refs(mach_task_self(), port_set, MACH_PORT_RIGHT_PORT_SET, -1);
	T_ASSERT_MACH_SUCCESS(ret, "mach_port_mod_refs(PORT_SET, -1)");

	ret = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_RECEIVE, -1);
	T_ASSERT_MACH_SUCCESS(ret, "mach_port_mod_refs(RECV_RIGHT, -1)");
}
