/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
 *	File:	mach/vm_behavior.h
 *
 *	Virtual memory map behavior definitions.
 *
 */

#ifndef _MACH_VM_BEHAVIOR_H_
#define _MACH_VM_BEHAVIOR_H_

/*
 *	Types defined:
 *
 *	vm_behavior_t	behavior codes.
 */

typedef int             vm_behavior_t;

/*
 *	Enumeration of valid values for vm_behavior_t.
 *	These describe expected page reference behavior for
 *	for a given range of virtual memory.  For implementation
 *	details see vm/vm_fault.c
 *
 *  When making a new VM_BEHAVIOR_*, update tests vm_parameter_validation_[user|kern]
 *  and their expected results; they deliberately call VM functions with invalid
 *  behavior values and you may be turning one of those invalid behaviors valid.
 */


/*
 * The following behaviors affect the memory region's future behavior
 * and are stored in the VM map entry data structure.
 */
#define VM_BEHAVIOR_DEFAULT     ((vm_behavior_t) 0)     /* default */
#define VM_BEHAVIOR_RANDOM      ((vm_behavior_t) 1)     /* random */
#define VM_BEHAVIOR_SEQUENTIAL  ((vm_behavior_t) 2)     /* forward sequential */
#define VM_BEHAVIOR_RSEQNTL     ((vm_behavior_t) 3)     /* reverse sequential */

/*
 * The following "behaviors" affect the memory region only at the time of the
 * call and are not stored in the VM map entry.
 */
#define VM_BEHAVIOR_WILLNEED    ((vm_behavior_t) 4)     /* will need in near future */
#define VM_BEHAVIOR_DONTNEED    ((vm_behavior_t) 5)     /* dont need in near future */
#define VM_BEHAVIOR_FREE        ((vm_behavior_t) 6)     /* free memory without write-back */
#define VM_BEHAVIOR_ZERO_WIRED_PAGES    ((vm_behavior_t) 7)     /* zero out the wired pages of an entry if it is being deleted without unwiring them first */
#define VM_BEHAVIOR_REUSABLE    ((vm_behavior_t) 8)
#define VM_BEHAVIOR_REUSE       ((vm_behavior_t) 9)
#define VM_BEHAVIOR_CAN_REUSE   ((vm_behavior_t) 10)
#define VM_BEHAVIOR_PAGEOUT     ((vm_behavior_t) 11)   /* force page-out of the pages in range (development only) */
#define VM_BEHAVIOR_ZERO        ((vm_behavior_t) 12)   /* zero pages without faulting in additional pages */

#define VM_BEHAVIOR_LAST_VALID (VM_BEHAVIOR_ZERO)

#endif  /*_MACH_VM_BEHAVIOR_H_*/
