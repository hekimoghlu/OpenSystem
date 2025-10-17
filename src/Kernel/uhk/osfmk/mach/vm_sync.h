/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
 *	File:	mach/vm_sync.h
 *
 *	Virtual memory synchronisation definitions.
 *
 */

#ifndef _MACH_VM_SYNC_H_
#define _MACH_VM_SYNC_H_

typedef unsigned                vm_sync_t;

/*
 *	Synchronization flags, defined as bits within the vm_sync_t type
 *
 *  When making a new VM_SYNC_*, update tests vm_parameter_validation_[user|kern]
 *  and their expected results; they deliberately call VM functions with invalid
 *  sync values and you may be turning one of those invalid syncs valid.
 */

#define VM_SYNC_ASYNCHRONOUS    ((vm_sync_t) 0x01)
#define VM_SYNC_SYNCHRONOUS     ((vm_sync_t) 0x02)
#define VM_SYNC_INVALIDATE      ((vm_sync_t) 0x04)
#define VM_SYNC_KILLPAGES       ((vm_sync_t) 0x08)
#define VM_SYNC_DEACTIVATE      ((vm_sync_t) 0x10)
#define VM_SYNC_CONTIGUOUS      ((vm_sync_t) 0x20)
#define VM_SYNC_REUSABLEPAGES   ((vm_sync_t) 0x40)

#endif  /* _MACH_VM_SYNC_H_ */
