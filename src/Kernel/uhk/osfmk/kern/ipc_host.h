/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
 */

#ifndef _KERN_IPC_HOST_H_
#define _KERN_IPC_HOST_H_

#include <mach/port.h>
#include <kern/kern_types.h>

/* Initialize IPC host services */
extern void ipc_host_init(void);

/* Initialize ipc access to processor by allocating a port */
extern void ipc_processor_init(
	processor_t     processor);

/* Initialize ipc control of a processor set */
extern void ipc_pset_init(
	processor_set_t         pset);

/* Initialize ipc control of a clock */
extern void ipc_clock_init(
	clock_t         clock);

/* Convert from a port to a clock */
extern clock_t convert_port_to_clock(
	ipc_port_t      port);

/* Convert from a clock to a port */
extern ipc_port_t convert_clock_to_port(
	clock_t         clock);

/* Convert from a clock name to a clock pointer */
extern clock_t port_name_to_clock(
	mach_port_name_t clock_name);

extern ipc_port_t host_port_copy_send(
	ipc_port_t      port);

/* Convert from a port to a host */
extern host_t convert_port_to_host(
	ipc_port_t      port);

/* Convert from a port to a host privilege port */
extern host_t convert_port_to_host_priv(
	ipc_port_t      port);

/* Convert from a host to a port */
extern ipc_port_t convert_host_to_port(
	host_t          host);

/* Convert from a port to a processor */
extern processor_t convert_port_to_processor(
	ipc_port_t      port);

/* Convert from a processor to a port */
extern ipc_port_t convert_processor_to_port(
	processor_t     processor);

/* Convert from a port to a processor set */
extern processor_set_t convert_port_to_pset(
	ipc_port_t      port);

/* Convert from a port to a processor set name */
extern processor_set_t convert_port_to_pset_name(
	ipc_port_t      port);

/* Convert from a processor set to a port */
extern ipc_port_t convert_pset_to_port(
	processor_set_t         processor);

/* Convert from a processor set name to a port */
extern ipc_port_t convert_pset_name_to_port(
	processor_set_t         processor);

#endif  /* _KERN_IPC_HOST_H_ */
