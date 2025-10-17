/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#ifdef  XNU_KERNEL_PRIVATE

#ifndef _KERN_AFFINITY_H_
#define _KERN_AFFINITY_H_

#ifdef  MACH_KERNEL_PRIVATE

#include <kern/queue.h>
#include <kern/processor.h>

/*
 * An affinity set object represents a set of threads identified by the user
 * to be sharing (cache) affinity. A task may have multiple affinity sets
 * defined. Each set has dis-affinity other sets. Tasks related by inheritance
 * may share the same affinity set namespace.
 * Affinity sets are used to advise (hint) thread placement.
 */
struct affinity_set {
	struct affinity_space *aset_space;      /* namespace */
	queue_chain_t   aset_affinities;        /* links affinities in group */
	queue_head_t    aset_threads;           /* threads in affinity set */
	uint32_t        aset_thread_count;      /* num threads in set */
	uint32_t        aset_tag;               /* user-assigned tag */
	uint32_t        aset_num;               /* kernel-assigned affinity */
	processor_set_t aset_pset;              /* processor set */
};

extern boolean_t        thread_affinity_is_supported(void);
extern void             thread_affinity_dup(thread_t parent, thread_t child);
extern void             thread_affinity_terminate(thread_t thread);
extern void             task_affinity_create(
	task_t,
	task_t);
extern void             task_affinity_deallocate(
	task_t);
extern kern_return_t    task_affinity_info(
	task_t,
	task_info_t,
	mach_msg_type_number_t  *);

#endif  /* MACH_KERNEL_PRIVATE */

extern kern_return_t    thread_affinity_set(thread_t thread, uint32_t tag);
extern uint32_t         thread_affinity_get(thread_t thread);
extern void             thread_affinity_exec(thread_t thread);

#endif  /* _KERN_AFFINITY_H_ */

#endif  /* XNU_KERNEL_PRIVATE */
