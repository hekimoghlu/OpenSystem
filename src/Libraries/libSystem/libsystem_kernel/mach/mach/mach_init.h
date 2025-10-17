/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
 * Mach Operating System
 * Copyright (c) 1991,1990,1989,1988,1987,1986 Carnegie Mellon University
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
 *	Items provided by the Mach environment initialization.
 */

#ifndef _MACH_INIT_
#define _MACH_INIT_     1

#include <mach/mach_types.h>
#include <mach/vm_page_size.h>
#include <stdarg.h>

#include <sys/cdefs.h>

#ifndef KERNEL
#include <Availability.h>
#endif

/*
 *	Kernel-related ports; how a task/thread controls itself
 */

__BEGIN_DECLS
extern mach_port_t mach_host_self(void);
extern mach_port_t mach_thread_self(void);
__API_AVAILABLE(macos(11.3), ios(14.5), tvos(14.5), watchos(7.3))
extern boolean_t mach_task_is_self(task_name_t task);
extern kern_return_t host_page_size(host_t, vm_size_t *);

extern mach_port_t      mach_task_self_;
#define mach_task_self() mach_task_self_
#define current_task()  mach_task_self()

__END_DECLS
#include <mach/mach_traps.h>
__BEGIN_DECLS

/*
 *	Other important ports in the Mach user environment
 */

extern  mach_port_t     bootstrap_port;

/*
 *	Where these ports occur in the "mach_ports_register"
 *	collection... only servers or the runtime library need know.
 */

#define NAME_SERVER_SLOT        0
#define ENVIRONMENT_SLOT        1
#define SERVICE_SLOT            2

#define MACH_PORTS_SLOTS_USED   3

/*
 *	fprintf_stderr uses vprintf_stderr_func to produce
 *	error messages, this can be overridden by a user
 *	application to point to a user-specified output function
 */
extern int (*vprintf_stderr_func)(const char *format, va_list ap);

__END_DECLS

#endif  /* _MACH_INIT_ */
