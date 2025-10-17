/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#ifndef _KERN_SCHED_URGENCY_H_
#define _KERN_SCHED_URGENCY_H_

#if defined(MACH_KERNEL_PRIVATE) || SCHED_TEST_HARNESS
__BEGIN_DECLS

#include <kern/kern_types.h>

typedef enum thread_urgency {
	THREAD_URGENCY_NONE             = 0,    /* processor is idle */
	THREAD_URGENCY_BACKGROUND       = 1,    /* "background" thread (i.e. min-power) */
	THREAD_URGENCY_NORMAL           = 2,    /* "normal" thread */
	THREAD_URGENCY_REAL_TIME        = 3,    /* "real-time" or urgent thread */
	THREAD_URGENCY_LOWPRI           = 4,    /* low priority but not "background" hint for performance management subsystem */
	THREAD_URGENCY_MAX              = 5,    /* Max */
} thread_urgency_t;

/* Returns the "urgency" of a thread (provided by scheduler) */
extern thread_urgency_t      thread_get_urgency(
	thread_t        thread,
	uint64_t        *rt_period,
	uint64_t        *rt_deadline);

/* Tells the "urgency" of the just scheduled thread (provided by CPU PM) */
extern void     thread_tell_urgency(
	thread_urgency_t             urgency,
	uint64_t        rt_period,
	uint64_t        rt_deadline,
	uint64_t        sched_latency,
	thread_t nthread);

__END_DECLS
#endif /* defined(MACH_KERNEL_PRIVATE) || SCHED_TEST_HARNESS */

#endif /* _KERN_SCHED_URGENCY_H_ */
