/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
 *	kern/extmod_statistics.h
 *
 *	Definitions for statistics related to external
 *  modification of a task by another agent on the system.
 *
 */

#ifndef _KERN_EXTMOD_STATISTICS_H_
#define _KERN_EXTMOD_STATISTICS_H_

#include <kern/task.h>
#include <mach/vm_types.h>

extern void extmod_statistics_incr_task_for_pid(task_t target);
extern void extmod_statistics_incr_thread_set_state(thread_t target);
extern void extmod_statistics_incr_thread_create(task_t target);

#endif  /* _KERN_EXTMOD_STATISTICS_H_ */
