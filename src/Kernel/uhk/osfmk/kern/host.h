/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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
/*
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989,1988 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */

/*
 *	kern/host.h
 *
 *	Definitions for host data structures.
 *
 */

#ifndef _KERN_HOST_H_
#define _KERN_HOST_H_

#include <mach/mach_types.h>
#include <sys/cdefs.h>

#ifdef  MACH_KERNEL_PRIVATE

#include <kern/locks.h>
#include <kern/exception.h>
#include <mach/exception_types.h>
#include <mach/host_special_ports.h>
#include <kern/kern_types.h>
#include <mach/vm_statistics.h>

struct  host {
	decl_lck_mtx_data(, lock);               /* lock to protect exceptions */
	ipc_port_t XNU_PTRAUTH_SIGNED_PTR("host.special") special[HOST_MAX_SPECIAL_PORT + 1];
	struct exception_action exc_actions[EXC_TYPES_COUNT];
};

typedef struct host     host_data_t;

extern host_data_t      realhost;

#define host_lock(host)         lck_mtx_lock(&(host)->lock)
#define host_unlock(host)       lck_mtx_unlock(&(host)->lock)

extern vm_extmod_statistics_data_t host_extmod_statistics;

typedef struct {
	uint64_t total_user_time;
	uint64_t total_system_time;
	uint64_t task_interrupt_wakeups;
	uint64_t task_platform_idle_wakeups;
	uint64_t task_timer_wakeups_bin_1;
	uint64_t task_timer_wakeups_bin_2;
	uint64_t total_ptime;
	uint64_t total_pset_switches;
	uint64_t task_gpu_ns;
	uint64_t task_energy;
} expired_task_statistics_t;

extern expired_task_statistics_t dead_task_statistics;

extern kern_return_t host_set_special_port(host_priv_t host_priv, int id, ipc_port_t port);
extern kern_return_t host_get_special_port(host_priv_t host_priv,
    __unused int node, int id, ipc_port_t * portp);

#endif  /* MACH_KERNEL_PRIVATE */

/*
 * Access routines for inside the kernel.
 */

__BEGIN_DECLS

extern host_t                   host_self(void);
extern host_priv_t              host_priv_self(void);

__END_DECLS

#endif  /* _KERN_HOST_H_ */
