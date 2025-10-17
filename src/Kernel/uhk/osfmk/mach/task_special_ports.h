/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
 * Copyright (c) 1991,1990,1989,1988,1987 Carnegie Mellon University
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
 */
/*
 *	File:	mach/task_special_ports.h
 *
 *	Defines codes for special_purpose task ports.  These are NOT
 *	port identifiers - they are only used for the task_get_special_port
 *	and task_set_special_port routines.
 *
 */

#ifndef _MACH_TASK_SPECIAL_PORTS_H_
#define _MACH_TASK_SPECIAL_PORTS_H_

typedef int     task_special_port_t;

#define TASK_KERNEL_PORT        1       /* The full task port for task. */

#define TASK_HOST_PORT          2       /* The host (priv) port for task.  */

#define TASK_NAME_PORT          3       /* The name port for task. */

#define TASK_BOOTSTRAP_PORT     4       /* Bootstrap environment for task. */

#define TASK_INSPECT_PORT       5       /* The inspect port for task. */

#define TASK_READ_PORT          6       /* The read port for task. */

/*
 * Evolving and likely to change.
 */

/* Was TASK_SEATBELT_PORT      7        Seatbelt compiler/DEM port for task. */

/* Was TASK_GSSD_PORT          8        which transformed to a host port */

#define TASK_ACCESS_PORT        9       /* Permission check for task_for_pid. */

#define TASK_DEBUG_CONTROL_PORT 10      /* debug control port */

#define TASK_RESOURCE_NOTIFY_PORT   11  /* overrides host special RN port */

#define TASK_MAX_SPECIAL_PORT TASK_RESOURCE_NOTIFY_PORT

/*
 *	Definitions for ease of use
 */

#define task_get_kernel_port(task, port)        \
	        (task_get_special_port((task), TASK_KERNEL_PORT, (port)))

#define task_set_kernel_port(task, port)        \
	        (task_set_special_port((task), TASK_KERNEL_PORT, (port)))

#define task_get_host_port(task, port)          \
	        (task_get_special_port((task), TASK_HOST_PORT, (port)))

#define task_set_host_port(task, port)  \
	        (task_set_special_port((task), TASK_HOST_PORT, (port)))

#define task_get_bootstrap_port(task, port)     \
	        (task_get_special_port((task), TASK_BOOTSTRAP_PORT, (port)))

#define task_get_debug_control_port(task, port) \
	        (task_get_special_port((task), TASK_DEBUG_CONTROL_PORT, (port)))

#define task_set_bootstrap_port(task, port)     \
	        (task_set_special_port((task), TASK_BOOTSTRAP_PORT, (port)))

#define task_get_task_access_port(task, port)   \
	        (task_get_special_port((task), TASK_ACCESS_PORT, (port)))

#define task_set_task_access_port(task, port)   \
	        (task_set_special_port((task), TASK_ACCESS_PORT, (port)))

#define task_set_task_debug_control_port(task, port) \
	        (task_set_special_port((task), TASK_DEBUG_CONTROL_PORT, (port)))

#ifdef XNU_KERNEL_PRIVATE
#define DEBUG_PORT_ENTITLEMENT "com.apple.private.debug_port"
#endif /* XNU_KERNEL_PRIVATE */

#endif  /* _MACH_TASK_SPECIAL_PORTS_H_ */
