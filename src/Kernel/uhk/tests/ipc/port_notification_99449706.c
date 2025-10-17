/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_TAG_VM_PREFERRED);

static mach_port_t
create_voucher(void)
{
	mach_voucher_attr_recipe_data_t dummy_voucher = {
		.key                = MACH_VOUCHER_ATTR_KEY_IMPORTANCE,
		.command            = MACH_VOUCHER_ATTR_IMPORTANCE_SELF,
		.previous_voucher   = MACH_VOUCHER_NULL,
		.content_size       = 0,
	};

	mach_port_t port = MACH_PORT_NULL;
	kern_return_t kr = host_create_mach_voucher(mach_host_self(),
	    (mach_voucher_attr_raw_recipe_array_t)&dummy_voucher,
	    sizeof(dummy_voucher), &port);
	T_ASSERT_MACH_SUCCESS(kr, "alloc voucher");

	return port;
}


T_DECL(mach_port_notification_dead_name_double_free, "Test mach_port_request_notification with a dead name port")
{
	kern_return_t kr;
	mach_port_t dead_port;
	mach_port_t voucher_port;

	kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_DEAD_NAME, &dead_port);
	T_ASSERT_MACH_SUCCESS(kr, "alloc dead port");

	voucher_port = create_voucher();
	T_ASSERT_NE(voucher_port, MACH_PORT_NULL, "voucher not null");

	/* trigger crash via double-free: see rdar://99779706 */
	mach_port_request_notification(mach_task_self(),
	    voucher_port,
	    MACH_NOTIFY_PORT_DESTROYED,
	    0, dead_port, MACH_MSG_TYPE_PORT_SEND_ONCE, 0);

	T_PASS("Kernel didn't crash!");
}
