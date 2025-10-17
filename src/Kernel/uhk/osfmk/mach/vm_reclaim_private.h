/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#pragma once
#if defined(__LP64__)
#include <mach/mach_types.h>
#include <mach/vm_reclaim.h>
#include <ptrcheck.h>

/*
 * This header exists for the internal implementation in libsyscall/xnu
 * and for observability with debugging tools. It should _NOT_ be used by
 * clients.
 */

#define VM_RECLAIM_MAX_BUFFER_SIZE (128ull << 20)
#define VM_RECLAIM_MAX_CAPACITY ((VM_RECLAIM_MAX_BUFFER_SIZE - \
	offsetof(struct mach_vm_reclaim_ring_s, entries)) / \
	sizeof(struct mach_vm_reclaim_entry_s))

__BEGIN_DECLS

typedef struct mach_vm_reclaim_indices_s {
	_Atomic mach_vm_reclaim_id_t head;
	_Atomic mach_vm_reclaim_id_t tail;
	_Atomic mach_vm_reclaim_id_t busy;
} *mach_vm_reclaim_indices_t;

typedef struct mach_vm_reclaim_entry_s {
	mach_vm_address_t address;
	uint32_t size;
	mach_vm_reclaim_action_t behavior;
	uint8_t _unused[3];
} *mach_vm_reclaim_entry_t;

/*
 * Contains the data used for synchronization with the kernel. This structure
 * should be page-aligned.
 */
struct mach_vm_reclaim_ring_s {
	mach_vm_size_t va_in_buffer;
	mach_vm_size_t last_accounting_given_to_kernel;
	mach_vm_reclaim_count_t len;
	mach_vm_reclaim_count_t max_len;
	struct mach_vm_reclaim_indices_s indices;
	/*
	 * The ringbuffer entries themselves populate the remainder of this
	 * buffer's vm allocation.
	 * NB: the fields preceding `entries` must be aligned to a multiple of
	 * the entry size.
	 */
	struct mach_vm_reclaim_entry_s entries[] __counted_by(len);
};

__END_DECLS
#endif /* __LP64__ */
