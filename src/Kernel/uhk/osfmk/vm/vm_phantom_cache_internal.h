/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include <vm/vm_page.h>

#define         VM_GHOST_OFFSET_BITS    39
#define         VM_GHOST_OFFSET_MASK    0x7FFFFFFFFF
#define         VM_GHOST_PAGES_PER_ENTRY 4
#define         VM_GHOST_PAGE_MASK      0x3
#define         VM_GHOST_PAGE_SHIFT     2
#define         VM_GHOST_INDEX_BITS     (64 - VM_GHOST_OFFSET_BITS - VM_GHOST_PAGES_PER_ENTRY)

struct  vm_ghost {
	uint64_t        g_next_index:VM_GHOST_INDEX_BITS,
	    g_pages_held:VM_GHOST_PAGES_PER_ENTRY,
	    g_obj_offset:VM_GHOST_OFFSET_BITS;
	uint32_t        g_obj_id;
} __attribute__((packed));

typedef struct vm_ghost *vm_ghost_t;


extern  void            vm_phantom_cache_init(void);
extern  void            vm_phantom_cache_add_ghost(vm_page_t);
extern  vm_ghost_t      vm_phantom_cache_lookup_ghost(vm_page_t, uint32_t);
extern  void            vm_phantom_cache_update(vm_page_t);
extern  boolean_t       vm_phantom_cache_check_pressure(void);
extern  void            vm_phantom_cache_restart_sample(void);
