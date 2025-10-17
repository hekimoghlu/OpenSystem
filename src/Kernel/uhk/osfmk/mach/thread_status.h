/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
/*
 *	File:	mach/thread_status.h
 *	Author:	Avadis Tevanian, Jr.
 *
 *	This file contains the structure definitions for the user-visible
 *	thread state.  This thread state is examined with the thread_get_state
 *	kernel call and may be changed with the thread_set_state kernel call.
 *
 */

#ifndef _MACH_THREAD_STATUS_H_
#define _MACH_THREAD_STATUS_H_

/*
 *	The actual structure that comprises the thread state is defined
 *	in the machine dependent module.
 */
#include <mach/machine/vm_types.h>
#include <mach/machine/thread_status.h>
#include <mach/machine/thread_state.h>

/*
 *	Generic definition for machine-dependent thread status.
 */

typedef natural_t       *thread_state_t;        /* Variable-length array */

/* THREAD_STATE_MAX is now defined in <mach/machine/thread_state.h> */
typedef natural_t       thread_state_data_t[THREAD_STATE_MAX];

#define THREAD_STATE_FLAVOR_LIST        0       /* List of valid flavors */
#define THREAD_STATE_FLAVOR_LIST_NEW    128
#define THREAD_STATE_FLAVOR_LIST_10_9   129
#define THREAD_STATE_FLAVOR_LIST_10_13  130
#define THREAD_STATE_FLAVOR_LIST_10_15  131

typedef int                     thread_state_flavor_t;
typedef thread_state_flavor_t   *thread_state_flavor_array_t;

#define THREAD_CONVERT_THREAD_STATE_TO_SELF   1
#define THREAD_CONVERT_THREAD_STATE_FROM_SELF 2

#endif  /* _MACH_THREAD_STATUS_H_ */
