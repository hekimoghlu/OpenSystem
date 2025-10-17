/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include <mach/host_priv.h>
#include <mach/host_special_ports.h>
#include <mach/mach_types.h>
#include <ipc/ipc_port.h>

#include <mach/sysdiagnose_notification.h>

#include <kern/misc_protos.h>
#include <kern/host.h>

#include <sys/kdebug.h>

extern kern_return_t sysdiagnose_notify_user(uint32_t);

/*
 * If userland has registered a port for sysdiagnose notifications, send one now.
 */
kern_return_t
sysdiagnose_notify_user(uint32_t keycode)
{
	mach_port_t user_port;
	kern_return_t kr;

	kr = host_get_sysdiagnose_port(host_priv_self(), &user_port);
	if ((kr != KERN_SUCCESS) || !IPC_PORT_VALID(user_port)) {
		return kr;
	}

	KERNEL_DEBUG_CONSTANT(MACHDBG_CODE(DBG_MACH_SYSDIAGNOSE, SYSDIAGNOSE_NOTIFY_USER) | DBG_FUNC_START, 0, 0, 0, 0, 0);

	kr = send_sysdiagnose_notification_with_audit_token(user_port, keycode);
	ipc_port_release_send(user_port);
	return kr;
}
