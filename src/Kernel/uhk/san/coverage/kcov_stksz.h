/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#ifndef _KCOV_STKSZ_H_
#define _KCOV_STKSZ_H_

#include <stdbool.h>

#include <kern/thread.h>
#include <mach/vm_types.h>

#include <san/kcov_stksz_data.h>

#if KERNEL_PRIVATE

#if CONFIG_STKSZ

__BEGIN_DECLS

void kcov_stksz_init_thread(kcov_stksz_thread_t *);
void kcov_stksz_update_stack_size(thread_t, kcov_thread_data_t *, void *, uintptr_t);

/* Sets ksancov stack for given thread. */
void kcov_stksz_set_thread_stack(thread_t, vm_offset_t);

/* Returns stack info for given thread. */
vm_offset_t kcov_stksz_get_thread_stkbase(thread_t);
vm_offset_t kcov_stksz_get_thread_stksize(thread_t);

__END_DECLS

#else

#define kcov_stksz_init_thread(thread)
#define kcov_stksz_update_stack_size(thread, data, caller, sp)

#endif /* CONFIG_STKSZ */

#endif /* KERNEL_PRIVATE */

#endif /* _KCOV_STKSZ_H_ */
