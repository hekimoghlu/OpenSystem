/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#ifndef _KERN_THREAD_KERNEL_STATE_H_

#include <vm/vm_kern.h>

struct thread_kernel_state {
	machine_thread_kernel_state  machine;       /* must be first */
	kern_allocation_name_t       allocation_name;
} __attribute__((aligned(16)));

typedef struct thread_kernel_state * thread_kernel_state_t;

#define thread_get_kernel_state(thread) ((thread_kernel_state_t) \
    ((thread)->kernel_stack + kernel_stack_size - sizeof(struct thread_kernel_state)))

#define thread_initialize_kernel_state(thread)  \
    thread_get_kernel_state((thread))->allocation_name = NULL;

#endif /* _KERN_THREAD_KERNEL_STATE_H_ */
