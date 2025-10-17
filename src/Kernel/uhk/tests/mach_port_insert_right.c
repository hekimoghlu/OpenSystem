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
#include <stdio.h>
#include <mach/mach.h>
#include <mach/message.h>
#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RUN_CONCURRENTLY(TRUE),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_TAG_VM_PREFERRED);

static inline mach_port_type_t
get_port_type(mach_port_t mp)
{
	mach_port_type_t type;
	T_QUIET;
	T_ASSERT_MACH_SUCCESS(mach_port_type(mach_task_self(), mp, &type),
	    "mach_port_type(mP)");
	return type;
}

T_DECL(mach_port_insert_right, "insert send right for an existing right", T_META_CHECK_LEAKS(false))
{
	mach_port_t port = MACH_PORT_NULL;
	mach_port_t port2 = MACH_PORT_NULL;
	kern_return_t retval;

	mach_port_t task = mach_task_self();

	retval = mach_port_allocate(task, MACH_PORT_RIGHT_RECEIVE, &port);
	T_ASSERT_MACH_SUCCESS(retval, "allocate a port=[%d]", port);

	T_ASSERT_EQ(get_port_type(port), MACH_PORT_TYPE_RECEIVE,
	    "0x%x should be a receive right", port);

	retval = mach_port_insert_right(task, port, port, MACH_MSG_TYPE_MAKE_SEND);
	T_ASSERT_MACH_SUCCESS(retval, "insert a send right for port=[%d] with name=[%d]", port, port);
	T_ASSERT_EQ(get_port_type(port), MACH_PORT_TYPE_RECEIVE | MACH_PORT_TYPE_SEND,
	    "0x%x should be a send-receive right", port);

	mach_port_name_t name = 123;
	/* 0x7B, lower two bits are 0b11 which is valid gen number, but index is invalid (0) */
	retval = mach_port_insert_right(task, name, port, MACH_MSG_TYPE_MAKE_SEND);
	T_ASSERT_MACH_ERROR(retval, KERN_FAILURE, "insert a send right for port=[%d] with name=[%d]", port, name);

	name = 122;
	/* invalid gen number, should return KERN_INVALID_VALUE */
	retval = mach_port_insert_right(task, name, port, MACH_MSG_TYPE_MAKE_SEND);
	T_ASSERT_MACH_ERROR(retval, KERN_INVALID_VALUE, "insert a send right for port=[%d] with name=[%d]", port, name);

	retval = mach_port_allocate(task, MACH_PORT_RIGHT_RECEIVE, &port2);
	T_ASSERT_MACH_SUCCESS(retval, "allocate a port=[%d]", port2);

	T_ASSERT_EQ(get_port_type(port2), MACH_PORT_TYPE_RECEIVE,
	    "0x%x should be a receive right", port2);

	name = port;
	retval = mach_port_insert_right(task, name, port2, MACH_MSG_TYPE_MAKE_SEND);
	T_ASSERT_MACH_ERROR(retval, KERN_RIGHT_EXISTS, "insert a send right for port=[%d] with name=[%d]", port2, name);
}
