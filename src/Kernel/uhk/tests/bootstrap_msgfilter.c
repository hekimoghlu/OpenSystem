/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include <mach/message.h>
#include <mach/mach_vm.h>
#include <mach/mach_port.h>
#include <mach/mach_error.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <darwintest.h>
#include <darwintest_utils.h>

#include <xpc/private.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("IPC"),
    T_META_NAMESPACE("xnu.ipc"),
    T_META_TAG_VM_PREFERRED);

struct test_msg {
	mach_msg_header_t header;
	mach_msg_trailer_t trailer; // subtract this when sending
};

T_DECL(bootstrap_mig_always_filtered,
    "'MIG' messages to bootstrap ports from tasks with filtering should always be filtered",
    T_META_ASROOT(true), T_META_REQUIRES_SYSCTL_EQ("kern.development", 1))
{
	int new_filter_flag = 1;
	int rc = sysctlbyname("kern.task_set_filter_msg_flag", NULL, NULL,
	    &new_filter_flag, sizeof(new_filter_flag));
	T_ASSERT_POSIX_SUCCESS(rc, "sysctlbyname");

	struct mach_service_port_info mspi = {
		.mspi_domain_type = XPC_DOMAIN_PORT,
	};
	strlcpy(mspi.mspi_string_name, "com.apple.xnu.test_bootstrap_msgfilter",
	    sizeof(mspi.mspi_string_name));

	mach_port_options_t port_opts = {
		.flags = MPO_SERVICE_PORT |
	    MPO_INSERT_SEND_RIGHT |
	    MPO_CONTEXT_AS_GUARD |
	    MPO_STRICT,
		.service_port_info = &mspi,
	};

	int ctxobj = 0;

	mach_port_t test_bootstrap_port;
	kern_return_t kr = mach_port_construct(mach_task_self(), &port_opts,
	    (uintptr_t)&ctxobj, &test_bootstrap_port);
	T_ASSERT_MACH_SUCCESS(kr, "mach_port_construct");

	// sending a valid 'XPC' msgid should succeed

	mach_msg_id_t permitted_xpc_msgid = 0x01000042;

	struct test_msg msg = {
		.header = {
			.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0),
			.msgh_size = offsetof(struct test_msg, trailer),
			.msgh_remote_port = test_bootstrap_port,
			.msgh_id = permitted_xpc_msgid,
		},
	};

	mach_msg_option_t msg_opts = MACH_SEND_MSG | MACH_RCV_MSG;
	kr = mach_msg(&msg.header, msg_opts, msg.header.msgh_size, sizeof(msg),
	    test_bootstrap_port, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
	T_ASSERT_MACH_SUCCESS(kr, "send message with valid (XPC) message ID");

	// sending a 'MIG' msgid (0x00xxxxxx) should fail, non-fatally

	mach_msg_id_t disallowed_mig_msgid = 0x00000042;
	msg_opts |= MACH_SEND_FILTER_NONFATAL;

	msg = (struct test_msg){
		.header = {
			.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0),
			.msgh_size = offsetof(struct test_msg, trailer),
			.msgh_remote_port = test_bootstrap_port,
			.msgh_id = disallowed_mig_msgid,
		},
	};
	kr = mach_msg(&msg.header, msg_opts, msg.header.msgh_size, sizeof(msg),
	    test_bootstrap_port, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
	T_ASSERT_EQ(kr, MACH_SEND_MSG_FILTERED, "message should be filtered");
}
