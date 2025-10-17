/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
 * Virtual memory map purgeable object definitions.
 * Objects that will be needed in the future (forward cached objects) should be queued LIFO.
 * Objects that have been used and are cached for reuse (backward cached) should be queued FIFO.
 * Every user of purgeable memory is entitled to using the highest volatile group (7).
 * Only if a client wants some of its objects to definitely be purged earlier, it can put those in
 * another group. This could be used to make all FIFO objects (in the lower group) go away before
 * any LIFO objects (in the higher group) go away.
 * Objects that should not get any chance to stay around can be marked as "obsolete". They will
 * be emptied before any other objects or pages are reclaimed. Obsolete objects are not emptied
 * in any particular order.
 * 'purgeable' is recognized as the correct spelling. For historical reasons, definitions
 * in this file are spelled 'purgable'.
 */

#ifndef _MACH_VM_PURGABLE_H_
#define _MACH_VM_PURGABLE_H_

/*
 *	Types defined:
 *
 *	vm_purgable_t	purgeable object control codes.
 */

typedef int     vm_purgable_t;

/*
 *	Enumeration of valid values for vm_purgable_t.
 *
 *  When making a new VM_PURGABLE_*, update tests vm_parameter_validation_[user|kern]
 *  and their expected results; they deliberately call VM functions with invalid
 *  values and you may be turning one of those invalid bits valid.
 */
#define VM_PURGABLE_SET_STATE   ((vm_purgable_t) 0)     /* set state of purgeable object */
#define VM_PURGABLE_GET_STATE   ((vm_purgable_t) 1)     /* get state of purgeable object */
#define VM_PURGABLE_PURGE_ALL   ((vm_purgable_t) 2)     /* purge all volatile objects now */
#define VM_PURGABLE_SET_STATE_FROM_KERNEL ((vm_purgable_t) 3) /* set state from kernel */

/*
 * Purgeable state:
 *
 *  31 15 14 13 12 11 10 8 7 6 5 4 3 2 1 0
 * +-----+--+-----+--+----+-+-+---+---+---+
 * |     |NA|DEBUG|  | GRP| |B|ORD|   |STA|
 * +-----+--+-----+--+----+-+-+---+---+---+
 * " ": unused (i.e. reserved)
 * STA: purgeable state
 *      see: VM_PURGABLE_NONVOLATILE=0 to VM_PURGABLE_DENY=3
 * ORD: order
 *      see:VM_VOLATILE_ORDER_*
 * B: behavior
 *      see: VM_PURGABLE_BEHAVIOR_*
 * GRP: group
 *      see: VM_VOLATILE_GROUP_*
 * DEBUG: debug
 *      see: VM_PURGABLE_DEBUG_*
 * NA: no aging
 *      see: VM_PURGABLE_NO_AGING*
 */

#define VM_PURGABLE_NO_AGING_SHIFT      16
#define VM_PURGABLE_NO_AGING_MASK       (0x1 << VM_PURGABLE_NO_AGING_SHIFT)
#define VM_PURGABLE_NO_AGING            (0x1 << VM_PURGABLE_NO_AGING_SHIFT)

#define VM_PURGABLE_DEBUG_SHIFT 12
#define VM_PURGABLE_DEBUG_MASK  (0x3 << VM_PURGABLE_DEBUG_SHIFT)
#define VM_PURGABLE_DEBUG_EMPTY (0x1 << VM_PURGABLE_DEBUG_SHIFT)
#define VM_PURGABLE_DEBUG_FAULT (0x2 << VM_PURGABLE_DEBUG_SHIFT)

/*
 * Volatile memory ordering groups (group zero objects are purged before group 1, etc...
 * It is implementation dependent as to whether these groups are global or per-address space.
 * (for the moment, they are global).
 */
#define VM_VOLATILE_GROUP_SHIFT         8
#define VM_VOLATILE_GROUP_MASK          (7 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_DEFAULT   VM_VOLATILE_GROUP_0

#define VM_VOLATILE_GROUP_0                     (0 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_1                     (1 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_2                     (2 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_3                     (3 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_4                     (4 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_5                     (5 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_6                     (6 << VM_VOLATILE_GROUP_SHIFT)
#define VM_VOLATILE_GROUP_7                     (7 << VM_VOLATILE_GROUP_SHIFT)

/*
 * Purgeable behavior
 * Within the same group, FIFO objects will be emptied before objects that are added later.
 * LIFO objects will be emptied after objects that are added later.
 * - Input only, not returned on state queries.
 */
#define VM_PURGABLE_BEHAVIOR_SHIFT  6
#define VM_PURGABLE_BEHAVIOR_MASK   (1 << VM_PURGABLE_BEHAVIOR_SHIFT)
#define VM_PURGABLE_BEHAVIOR_FIFO   (0 << VM_PURGABLE_BEHAVIOR_SHIFT)
#define VM_PURGABLE_BEHAVIOR_LIFO   (1 << VM_PURGABLE_BEHAVIOR_SHIFT)

/*
 * Obsolete object.
 * Disregard volatile group, and put object into obsolete queue instead, so it is the next object
 * to be purged.
 * - Input only, not returned on state queries.
 */
#define VM_PURGABLE_ORDERING_SHIFT              5
#define VM_PURGABLE_ORDERING_MASK               (1 << VM_PURGABLE_ORDERING_SHIFT)
#define VM_PURGABLE_ORDERING_OBSOLETE   (1 << VM_PURGABLE_ORDERING_SHIFT)
#define VM_PURGABLE_ORDERING_NORMAL             (0 << VM_PURGABLE_ORDERING_SHIFT)


/*
 * Obsolete parameter - do not use
 */
#define VM_VOLATILE_ORDER_SHIFT                 4
#define VM_VOLATILE_ORDER_MASK                  (1 << VM_VOLATILE_ORDER_SHIFT)
#define VM_VOLATILE_MAKE_FIRST_IN_GROUP (1 << VM_VOLATILE_ORDER_SHIFT)
#define VM_VOLATILE_MAKE_LAST_IN_GROUP  (0 << VM_VOLATILE_ORDER_SHIFT)

/*
 * Valid states of a purgeable object.
 */
#define VM_PURGABLE_STATE_MIN   0               /* minimum purgeable object state value */
#define VM_PURGABLE_STATE_MAX   3               /* maximum purgeable object state value */
#define VM_PURGABLE_STATE_MASK  3               /* mask to separate state from group */

#define VM_PURGABLE_NONVOLATILE 0               /* purgeable object is non-volatile */
#define VM_PURGABLE_VOLATILE    1               /* purgeable object is volatile */
#define VM_PURGABLE_EMPTY       2               /* purgeable object is volatile and empty */
#define VM_PURGABLE_DENY        3               /* (mark) object not purgeable */

#define VM_PURGABLE_ALL_MASKS   (VM_PURGABLE_STATE_MASK | \
	                         VM_VOLATILE_ORDER_MASK | \
	                         VM_PURGABLE_ORDERING_MASK | \
	                         VM_PURGABLE_BEHAVIOR_MASK | \
	                         VM_VOLATILE_GROUP_MASK | \
	                         VM_PURGABLE_DEBUG_MASK | \
	                         VM_PURGABLE_NO_AGING_MASK)
#endif  /* _MACH_VM_PURGABLE_H_ */
