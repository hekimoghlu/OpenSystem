/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#include <errno.h>
#include <mach/host_special_ports.h>
#include <mach/task_special_ports.h>
#include <mach/thread_special_ports.h>
#include <mach/port_descriptions.h>
#include <stdlib.h>
#include <strings.h>

const char *
mach_host_special_port_description(int port)
{
	int port_index = (int)port;

	if (port_index < 0 || port_index > HOST_MAX_SPECIAL_PORT) {
		return NULL;
	}

	static const char *hsp_descs[] = {
		[HOST_PORT] = "host (restricted)",
		[HOST_PRIV_PORT] = "host private (restricted)",
		[HOST_IO_MASTER_PORT] = "I/O master (restricted)",

		[HOST_DYNAMIC_PAGER_PORT] = "dynamic pager",
		[HOST_AUDIT_CONTROL_PORT] = "audit control",
		[HOST_USER_NOTIFICATION_PORT] = "user notification",
		[HOST_AUTOMOUNTD_PORT] = "automounter",
		[HOST_LOCKD_PORT] = "lockd",
		[HOST_KTRACE_BACKGROUND_PORT] = "ktrace background notification",
		[HOST_SEATBELT_PORT] = "seatbelt",
		[HOST_KEXTD_PORT] = "kextd",
		[HOST_LAUNCHCTL_PORT] = "launchctl",
		[HOST_UNFREED_PORT] = "fairplay",
		[HOST_AMFID_PORT] = "amfi",
		[HOST_GSSD_PORT] = "gssd",
		[HOST_TELEMETRY_PORT] = "telemetry",
		[HOST_ATM_NOTIFICATION_PORT] = "atm notification",
		[HOST_COALITION_PORT] = "coalition notification",
		[HOST_SYSDIAGNOSE_PORT] = "sysdiagnose notification",
		[HOST_XPC_EXCEPTION_PORT] = "XPC exception",
		[HOST_CONTAINERD_PORT] = "container manager",
		[HOST_NODE_PORT] = "node",
		[HOST_RESOURCE_NOTIFY_PORT] = "resource notify",
		[HOST_CLOSURED_PORT] = "closured",
		[HOST_SYSPOLICYD_PORT] = "syspolicyd",
		[HOST_FILECOORDINATIOND_PORT] = "filecoordinationd",
		[HOST_FAIRPLAYD_PORT] = "fairplayd",
		[HOST_IOCOMPRESSIONSTATS_PORT] = "I/O compression stats",
	};
	_Static_assert(HOST_IOCOMPRESSIONSTATS_PORT == HOST_MAX_SPECIAL_PORT,
	    "all host special ports must have descriptions");

	return hsp_descs[port_index];
}

const char *
mach_task_special_port_description(int port)
{
	int port_index = (int)port;

	if (port_index < 0 || port_index > TASK_MAX_SPECIAL_PORT) {
		return NULL;
	}

	static const char *tsp_descs[] = {
		[TASK_KERNEL_PORT] = "kernel",
		[TASK_HOST_PORT] = "host",
		[TASK_NAME_PORT] = "name",
		[TASK_BOOTSTRAP_PORT] = "bootstrap",
		[TASK_INSPECT_PORT] = "inspect",
		[TASK_READ_PORT] = "read",
		[TASK_SEATBELT_PORT] = "seatbelt",
		[TASK_ACCESS_PORT] = "access",
		[TASK_DEBUG_CONTROL_PORT] = "debug control",
		[TASK_RESOURCE_NOTIFY_PORT] = "resource notify",
	};
	_Static_assert(TASK_RESOURCE_NOTIFY_PORT == TASK_MAX_SPECIAL_PORT,
	    "all task special ports must have descriptions");

	return tsp_descs[port_index];
}

const char *
mach_thread_special_port_description(int port)
{
	int port_index = (int)port;

	if (port_index < 0 || port_index > THREAD_MAX_SPECIAL_PORT) {
		return NULL;
	}

	static const char *tsp_descs[] = {
		[THREAD_KERNEL_PORT] = "kernel",
		[THREAD_INSPECT_PORT] = "inspect",
		[THREAD_READ_PORT] = "read",
	};
	_Static_assert(THREAD_READ_PORT == THREAD_MAX_SPECIAL_PORT,
	    "all thread special ports must have descriptions");

	return tsp_descs[port_index];
}

static int
port_for_id_internal(const char *id, const char **ids, int nids)
{
	if (!id) {
		errno = EINVAL;
		return -1;
	}

	for (int i = 0; i < nids; i++) {
		if (ids[i] && strcmp(ids[i], id) == 0) {
			return i;
		}
	}

	errno = ENOENT;
	return -1;
}

int
mach_host_special_port_for_id(const char *id)
{
	static const char *hsp_ids[] = {
#define SP_ENTRY(id) [id] = #id
		SP_ENTRY(HOST_PORT),
		SP_ENTRY(HOST_PRIV_PORT),
		SP_ENTRY(HOST_IO_MASTER_PORT),
		SP_ENTRY(HOST_DYNAMIC_PAGER_PORT),
		SP_ENTRY(HOST_AUDIT_CONTROL_PORT),
		SP_ENTRY(HOST_USER_NOTIFICATION_PORT),
		SP_ENTRY(HOST_AUTOMOUNTD_PORT),
		SP_ENTRY(HOST_LOCKD_PORT),
		SP_ENTRY(HOST_KTRACE_BACKGROUND_PORT),
		SP_ENTRY(HOST_SEATBELT_PORT),
		SP_ENTRY(HOST_KEXTD_PORT),
		SP_ENTRY(HOST_LAUNCHCTL_PORT),
		SP_ENTRY(HOST_UNFREED_PORT),
		SP_ENTRY(HOST_AMFID_PORT),
		SP_ENTRY(HOST_GSSD_PORT),
		SP_ENTRY(HOST_TELEMETRY_PORT),
		SP_ENTRY(HOST_ATM_NOTIFICATION_PORT),
		SP_ENTRY(HOST_COALITION_PORT),
		SP_ENTRY(HOST_SYSDIAGNOSE_PORT),
		SP_ENTRY(HOST_XPC_EXCEPTION_PORT),
		SP_ENTRY(HOST_CONTAINERD_PORT),
		SP_ENTRY(HOST_NODE_PORT),
		SP_ENTRY(HOST_RESOURCE_NOTIFY_PORT),
		SP_ENTRY(HOST_CLOSURED_PORT),
		SP_ENTRY(HOST_SYSPOLICYD_PORT),
		SP_ENTRY(HOST_FILECOORDINATIOND_PORT),
	};

	return port_for_id_internal(id, hsp_ids,
	           sizeof(hsp_ids) / sizeof(hsp_ids[0]));
}

int
mach_task_special_port_for_id(const char *id)
{
	static const char *tsp_ids[] = {
		SP_ENTRY(TASK_KERNEL_PORT),
		SP_ENTRY(TASK_HOST_PORT),
		SP_ENTRY(TASK_NAME_PORT),
		SP_ENTRY(TASK_BOOTSTRAP_PORT),
		SP_ENTRY(TASK_INSPECT_PORT),
		SP_ENTRY(TASK_READ_PORT),
		SP_ENTRY(TASK_SEATBELT_PORT),
		SP_ENTRY(TASK_ACCESS_PORT),
		SP_ENTRY(TASK_DEBUG_CONTROL_PORT),
		SP_ENTRY(TASK_RESOURCE_NOTIFY_PORT),
	};

	return port_for_id_internal(id, tsp_ids,
	           sizeof(tsp_ids) / sizeof(tsp_ids[0]));
}

int
mach_thread_special_port_for_id(const char *id)
{
	static const char *tsp_ids[] = {
		SP_ENTRY(THREAD_KERNEL_PORT),
		SP_ENTRY(THREAD_INSPECT_PORT),
		SP_ENTRY(THREAD_READ_PORT),
#undef SP_ENTRY
	};

	return port_for_id_internal(id, tsp_ids,
	           sizeof(tsp_ids) / sizeof(tsp_ids[0]));
}
