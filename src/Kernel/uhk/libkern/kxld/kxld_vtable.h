/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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
#ifndef _KXLD_VTABLE_H_
#define _KXLD_VTABLE_H_

#include <sys/types.h>
#if KERNEL
    #include <libkern/kxld_types.h>
#else
    #include "kxld_types.h"
#endif

#include "kxld_array.h"

struct kxld_array;
struct kxld_object;
struct kxld_reloc;
struct kxld_relocator;
struct kxld_sect;
struct kxld_sym;
struct kxld_symtab;
struct kxld_vtable_hdr;
struct section;

typedef struct kxld_vtable KXLDVTable;
typedef union kxld_vtable_entry KXLDVTableEntry;

struct kxld_vtable {
	u_char *vtable;
	const char *name;
	KXLDArray entries;
	boolean_t is_patched;
};

struct kxld_vtable_patched_entry {
	char *name;
	kxld_addr_t addr;
};

struct kxld_vtable_unpatched_entry {
	const struct kxld_sym *sym;
	struct kxld_reloc *reloc;
};

union kxld_vtable_entry {
	struct kxld_vtable_patched_entry patched;
	struct kxld_vtable_unpatched_entry unpatched;
};

/*******************************************************************************
* Constructors and destructors
*******************************************************************************/

kern_return_t kxld_vtable_init(KXLDVTable *vtable,
    const struct kxld_sym *vtable_sym, const struct kxld_object *object,
    const struct kxld_dict *defined_cxx_symbols)
__attribute__((nonnull, visibility("hidden")));

void kxld_vtable_clear(KXLDVTable *vtable)
__attribute__((visibility("hidden")));

void kxld_vtable_deinit(KXLDVTable *vtable)
__attribute__((visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

KXLDVTableEntry * kxld_vtable_get_entry_for_offset(const KXLDVTable *vtable,
    u_long offset, boolean_t is_32_bit)
__attribute__((pure, nonnull, visibility("hidden")));

/*******************************************************************************
* Modifiers
*******************************************************************************/

/* With strict patching, the vtable patcher with only patch pad slots */
kern_return_t kxld_vtable_patch(KXLDVTable *vtable, const KXLDVTable *super_vtable,
    struct kxld_object *object)
__attribute__((nonnull, visibility("hidden")));

#endif /* _KXLD_VTABLE_H_ */
