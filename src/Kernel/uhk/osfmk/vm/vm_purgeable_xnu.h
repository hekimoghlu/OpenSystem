/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
 * Purgeable spelling rules
 * It is believed that the correct spelling is
 * { 'p', 'u', 'r', 'g', 'e', 'a', 'b', 'l', 'e' }.
 * However, there is one published API that likes to spell it without the
 * first 'e', vm_purgable_control(). Since we can't change that API,
 * here are the rules.
 * All qualifiers defined in vm_purgable.h are spelled without the e.
 * All other qualifiers are spelled with the e.
 * Right now, there are remains of the wrong spelling throughout the code,
 * vm_object_t.purgable for example. We expect to change these on occasion.
 */

#ifndef __VM_PURGEABLE_XNU__
#define __VM_PURGEABLE_XNU__

#ifdef XNU_KERNEL_PRIVATE

#include <kern/queue.h>

/* the object purger. purges the next eligible object from memory. */
/* returns TRUE if an object was purged, otherwise FALSE. */
boolean_t vm_purgeable_object_purge_one(int force_purge_below_group, int flags);

/* statistics for purgable objects in all queues */
void vm_purgeable_stats(vm_purgeable_info_t info, task_t target_task);

#if DEVELOPMENT || DEBUG
/* statistics for purgeable object usage in all queues for a task */
kern_return_t vm_purgeable_account(task_t task, pvm_account_info_t acnt_info);
#endif /* DEVELOPMENT || DEBUG */

uint64_t vm_purgeable_purge_task_owned(task_t task);


#endif /* XNU_KERNEL_PRIVATE */

#endif  /* __VM_PURGEABLE_XNU__ */
