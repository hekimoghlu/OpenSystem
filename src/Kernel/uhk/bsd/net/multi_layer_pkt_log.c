/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include <sys/sysctl.h>
#include <sys/proc.h>
#include <net/multi_layer_pkt_log.h>

SYSCTL_NODE(_net, OID_AUTO, mpklog,
    CTLFLAG_RW | CTLFLAG_LOCKED, 0, "Multi-layer packet logging");

/*
 * Note:  net_mpklog_enabled allows to override the interface flags IFXF_MPK_LOG
 */
int net_mpklog_enabled = 1;
static int sysctl_net_mpklog_enabled SYSCTL_HANDLER_ARGS;
SYSCTL_PROC(_net_mpklog, OID_AUTO, enabled, CTLTYPE_INT | CTLFLAG_LOCKED | CTLFLAG_RW,
    0, 0, &sysctl_net_mpklog_enabled, "I", "Multi-layer packet logging enabled");

static int sysctl_net_mpklog_type SYSCTL_HANDLER_ARGS;
uint8_t net_mpklog_type =  OS_LOG_TYPE_DEFAULT;
SYSCTL_PROC(_net_mpklog, OID_AUTO, type, CTLTYPE_INT | CTLFLAG_LOCKED | CTLFLAG_RW,
    0, 0, &sysctl_net_mpklog_type, "I", "Multi-layer packet logging type");

SYSCTL_INT(_net_mpklog, OID_AUTO, version, CTLFLAG_RD | CTLFLAG_LOCKED,
    (int *)NULL, MPKL_VERSION, "Multi-layer packet logging version");

static int
sysctl_net_mpklog_enabled SYSCTL_HANDLER_ARGS
{
#pragma unused(arg1, arg2)
	int value = net_mpklog_enabled;

	int error = sysctl_handle_int(oidp, &value, 0, req);
	if (error || !req->newptr) {
		return error;
	}

	net_mpklog_enabled = (value == 0) ? 0 : 1;

	os_log(OS_LOG_DEFAULT, "%s:%d set net_mpklog_enabled to %d",
	    proc_best_name(current_proc()), proc_selfpid(), net_mpklog_enabled);

	return 0;
}

static int
sysctl_net_mpklog_type SYSCTL_HANDLER_ARGS
{
#pragma unused(arg1, arg2)
	int value = net_mpklog_type;

	int error = sysctl_handle_int(oidp, &value, 0, req);
	if (error || !req->newptr) {
		return error;
	}

	if (value != OS_LOG_TYPE_DEFAULT &&
	    value != OS_LOG_TYPE_INFO) {
		return EINVAL;
	}
	net_mpklog_type = (uint8_t)value;

	os_log(OS_LOG_DEFAULT, "%s:%d set net_mpklog_type to %u (%s)",
	    proc_best_name(current_proc()), proc_selfpid(), net_mpklog_type,
	    net_mpklog_type == OS_LOG_TYPE_DEFAULT ? "default" : "info");

	return 0;
}
