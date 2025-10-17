/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

#ifndef _MACH_THREAD_SWITCH_H_
#define _MACH_THREAD_SWITCH_H_

#include <mach/mach_types.h>
#include <mach/kern_return.h>
#include <mach/message.h>
#include <mach/mach_traps.h>

/*
 *	Constant definitions for thread_switch trap.
 */

#define SWITCH_OPTION_NONE                      0
#define SWITCH_OPTION_DEPRESS           1
#define SWITCH_OPTION_WAIT                      2
#ifdef PRIVATE
/* Workqueue should not consider thread blocked, and option_time is in us */
#define SWITCH_OPTION_DISPATCH_CONTENTION       3
/* Handoff to lock owner and temporarily grant matching IO throttling policy */
#define SWITCH_OPTION_OSLOCK_DEPRESS    4
#define SWITCH_OPTION_OSLOCK_WAIT       5
#endif /* PRIVATE */

#define valid_switch_option(opt)        (0 <= (opt) && (opt) <= 5)

#endif  /* _MACH_THREAD_SWITCH_H_ */
