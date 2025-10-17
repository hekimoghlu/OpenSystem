/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#ifndef _KERN_STACKSHOT_H_
#define _KERN_STACKSHOT_H_

#include <stdint.h>
#include <kern/kern_types.h>
#include <kern/kern_cdata.h>

__BEGIN_DECLS

#ifdef XNU_KERNEL_PRIVATE

extern void                   kdp_snapshot_preflight(int pid, void * tracebuf, uint32_t tracebuf_size,
    uint64_t flags, kcdata_descriptor_t data_p, uint64_t since_timestamp, uint32_t pagetable_mask);
extern uint32_t               kdp_stack_snapshot_bytes_traced(void);
extern uint32_t               kdp_stack_snapshot_bytes_uncompressed(void);
extern boolean_t              stackshot_thread_is_idle_worker_unsafe(thread_t thread);
extern void                   stackshot_cpu_preflight(void);
extern void                   stackshot_aux_cpu_entry(void);
extern void                   stackshot_cpu_signal_panic(void);
extern kern_return_t          kern_stack_snapshot_internal(int stackshot_config_version, void *stackshot_config,
    size_t stackshot_config_size, boolean_t stackshot_from_user);
extern kern_return_t          do_stackshot(void* context);
extern boolean_t              stackshot_active(void);
extern boolean_t              panic_stackshot_active(void);
extern kern_return_t do_panic_stackshot(void *context);
extern void *                 stackshot_alloc_with_size(size_t size, kern_return_t *err);

/* Allocates an array of elements of a type from the stackshot buffer. Works in regular & panic stackshots. */
#define stackshot_alloc_arr(type, count, err) stackshot_alloc_with_size(sizeof(type) * (count), err)

/* Allocates an element with a type from the stackshot buffer. Works in regular & panic stackshot. */
#define stackshot_alloc(type, err) stackshot_alloc_with_size(sizeof(type), err)

#endif /* XNU_KERNEL_PRIVATE */

__END_DECLS

#endif /* _KERN_STACKSHOT_H_ */
