/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
#include <dlfcn.h>
#include <dlfcn_private.h>
#include <mach-o/dyld.h>
#include <dispatch/dispatch.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_RUN_CONCURRENTLY(TRUE));

T_DECL(task_dyld_process_info_notify_register,
    "check that task_dyld_process_info_notify_register works")
{
	mach_port_name_t port = MACH_PORT_NULL;
	dispatch_source_t ds;

	T_ASSERT_MACH_SUCCESS(mach_port_allocate(mach_task_self(),
	    MACH_PORT_RIGHT_RECEIVE, &port), "allocate notif port");

	ds = dispatch_source_create(DISPATCH_SOURCE_TYPE_MACH_RECV, port, 0,
	    dispatch_get_global_queue(0, 0));
	dispatch_source_set_event_handler(ds, ^{
		T_PASS("received a message for dlopen!");
		T_END;
	});
	dispatch_activate(ds);

	T_ASSERT_MACH_SUCCESS(task_dyld_process_info_notify_register(mach_task_self(), port),
	    "register dyld notification");

	dlopen("/usr/lib/swift/libswiftRemoteMirror.dylib", RTLD_LAZY);
}
