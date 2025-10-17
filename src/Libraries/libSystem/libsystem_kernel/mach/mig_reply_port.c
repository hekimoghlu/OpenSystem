/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <sys/cdefs.h>
#include <os/tsd.h>

__XNU_PRIVATE_EXTERN mach_port_t _task_reply_port = MACH_PORT_NULL;

static inline mach_port_t
_mig_get_reply_port()
{
	return (mach_port_t)(uintptr_t)_os_tsd_get_direct(__TSD_MIG_REPLY);
}

static inline void
_mig_set_reply_port(mach_port_t port)
{
	_os_tsd_set_direct(__TSD_MIG_REPLY, (void *)(uintptr_t)port);
}

/*
 * Called by mig interface code whenever a reply port is needed.
 * Tracing is masked during this call; otherwise, a call to printf()
 * can result in a call to malloc() which eventually reenters
 * mig_get_reply_port() and deadlocks.
 */
mach_port_t
mig_get_reply_port(void)
{
	mach_port_t port = _mig_get_reply_port();
	if (port == MACH_PORT_NULL) {
		port = mach_reply_port();
		_mig_set_reply_port(port);
	}
	return port;
}

/*
 * Called by mig interface code after a timeout on the reply port.
 * May also be called by user. The new mig calls with port passed in.
 */
void
mig_dealloc_reply_port(mach_port_t migport)
{
	mach_port_t port = _mig_get_reply_port();
	if (port != MACH_PORT_NULL && port != _task_reply_port) {
		_mig_set_reply_port(_task_reply_port);
		(void) mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_RECEIVE, -1);
		if (migport != port) {
			(void) mach_port_deallocate(mach_task_self(), migport);
		}
		_mig_set_reply_port(MACH_PORT_NULL);
	}
}

/*************************************************************
 *  Called by mig interfaces after each RPC.
 *  Could be called by user.
 ***********************************************************/

void
mig_put_reply_port(mach_port_t reply_port __unused)
{
}
