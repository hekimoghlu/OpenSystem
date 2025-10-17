/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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

#define NR_PORTS 4

T_DECL(mach_port_deallocate, "mach_port_deallocate deallocates also PORT_SET", T_META_TAG_VM_PREFERRED){
	mach_port_t port_set;
	mach_port_t port[NR_PORTS];
	int i, ret;

	ret = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_PORT_SET, &port_set);
	T_ASSERT_MACH_SUCCESS(ret, "mach_port_allocate MACH_PORT_RIGHT_PORT_SET");

	for (i = 0; i < NR_PORTS; i++) {
		ret = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port[i]);
		T_ASSERT_MACH_SUCCESS(ret, "mach_port_allocate MACH_PORT_RIGHT_RECEIVE");

		ret = mach_port_move_member(mach_task_self(), port[i], port_set);
		T_ASSERT_MACH_SUCCESS(ret, "mach_port_move_member");
	}

	T_LOG("Ports created");

	/* do something */

	for (i = 0; i < NR_PORTS; i++) {
		ret = mach_port_mod_refs(mach_task_self(), port[i], MACH_PORT_RIGHT_RECEIVE, -1);
		T_ASSERT_MACH_SUCCESS(ret, "mach_port_mod_refs -1 RIGHT_RECEIVE");
	}

	ret = mach_port_deallocate(mach_task_self(), port_set);
	T_ASSERT_MACH_SUCCESS(ret, "mach_port_deallocate PORT_SET");

	T_LOG("Ports erased");
}
