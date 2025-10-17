/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
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
 *	File:	ipc/ipc_port.h
 *	Author:	Rich Draves
 *	Date:	1989
 *
 *	Implementation specific complement to mach/port.h.
 */

#ifndef _IPC_PORT_H_
#define _IPC_PORT_H_

#include <mach/port.h>
#include <ipc/ipc_space.h>

#define MACH_PORT_NGEN(name)            MACH_PORT_MAKE(0, MACH_PORT_GEN(name))

/*
 *	Typedefs for code cleanliness.  These must all have
 *	the same (unsigned) type as mach_port_name_t.
 */


#define MACH_PORT_UREFS_MAX     ((mach_port_urefs_t) ((1 << 16) - 1))

#define MACH_PORT_UREFS_OVERFLOW(urefs, delta)                          \
	        (((delta) > 0) &&                                       \
	         ((((urefs) + (delta)) <= (urefs)) ||                   \
	          (((urefs) + (delta)) >= MACH_PORT_UREFS_MAX)))

#define MACH_PORT_UREFS_UNDERFLOW(urefs, delta)                         \
	        (((delta) < 0) && (((mach_port_urefs_t)-(delta)) > (urefs)))

#endif  /* _IPC_PORT_H_ */
