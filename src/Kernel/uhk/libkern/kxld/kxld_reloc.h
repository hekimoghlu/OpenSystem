/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef _KXLD_RELOC_H
#define _KXLD_RELOC_H

#include <mach/machine.h>
#include <sys/types.h>
#if KERNEL
    #include <libkern/kxld_types.h>
#else
    #include "kxld_types.h"
#endif

struct kxld_array;
struct kxld_dict;
struct kxld_sect;
struct kxld_seg;
struct kxld_sym;
struct kxld_symtab;
struct kxld_vtable;
struct relocation_info;

typedef struct kxld_relocator KXLDRelocator;
typedef struct kxld_reloc KXLDReloc;

typedef boolean_t (*RelocHasPair)(u_int r_type);
typedef u_int (*RelocGetPairType)(u_int prev_r_type);
typedef boolean_t (*RelocHasGot)(u_int r_type);
typedef kern_return_t (*ProcessReloc)(const KXLDRelocator *relocator,
    u_char *instruction, u_int length, u_int pcrel, kxld_addr_t base_pc,
    kxld_addr_t link_pc, kxld_addr_t link_disp, u_int type,
    kxld_addr_t target, kxld_addr_t pair_target, boolean_t swap);

struct kxld_relocator {
	RelocHasPair reloc_has_pair;
	RelocGetPairType reloc_get_pair_type;
	RelocHasGot reloc_has_got;
	ProcessReloc process_reloc;
	const struct kxld_symtab *symtab;
	const struct kxld_array *sectarray;
	const struct kxld_dict *vtables;
	const struct kxld_vtable *current_vtable;
	u_char *file;
	u_int function_align; /* Power of two alignment of functions */
	boolean_t is_32_bit;
	boolean_t swap;
	boolean_t may_scatter;
};

struct kxld_reloc {
	u_int address;
	u_int pair_address;
	u_int target;
	u_int pair_target;
	u_int target_type:3;
	u_int pair_target_type:3;
	u_int reloc_type:4;
	u_int length:2;
	u_int pcrel:1;
};

/*******************************************************************************
* Constructors and Destructors
*******************************************************************************/
kern_return_t kxld_relocator_init(KXLDRelocator *relocator, u_char *file,
    const struct kxld_symtab *symtab, const struct kxld_array *sectarray,
    cpu_type_t cputype, cpu_subtype_t cpusubtype, boolean_t swap)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_reloc_create_macho(struct kxld_array *relocarray,
    const KXLDRelocator *relocator, const struct relocation_info *srcs,
    u_int nsrcs) __attribute__((nonnull, visibility("hidden")));

void kxld_relocator_clear(KXLDRelocator *relocator)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

boolean_t kxld_relocator_has_pair(const KXLDRelocator *relocator, u_int r_type)
__attribute__((pure, nonnull, visibility("hidden")));

u_int kxld_relocator_get_pair_type(const KXLDRelocator *relocator,
    u_int last_r_type)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_relocator_has_got(const KXLDRelocator *relocator, u_int r_type)
__attribute__((pure, nonnull, visibility("hidden")));

kxld_addr_t kxld_relocator_get_pointer_at_addr(const KXLDRelocator *relocator,
    const u_char *data, u_long offset)
__attribute__((pure, nonnull, visibility("hidden")));

struct kxld_sym * kxld_reloc_get_symbol(const KXLDRelocator *relocator,
    const KXLDReloc *reloc, const u_char *data)
__attribute__((pure, nonnull(1, 2), visibility("hidden")));

kern_return_t kxld_reloc_get_reloc_index_by_offset(const struct kxld_array *relocs,
    kxld_size_t offset, u_int *idx)
__attribute__((nonnull, visibility("hidden")));

KXLDReloc * kxld_reloc_get_reloc_by_offset(const struct kxld_array *relocs,
    kxld_addr_t offset)
__attribute__((pure, nonnull, visibility("hidden")));

#if KXLD_PIC_KEXTS
u_long kxld_reloc_get_macho_header_size(void)
__attribute__((pure, visibility("hidden")));

u_long kxld_reloc_get_macho_data_size(const struct kxld_array *locrelocs,
    const struct kxld_array *extrelocs)
__attribute__((pure, nonnull, visibility("hidden")));

kern_return_t kxld_reloc_export_macho(const KXLDRelocator *relocator,
    const struct kxld_array *locrelocs, const struct kxld_array *extrelocs,
    u_char *buf, u_long *header_offset, u_long header_size,
    u_long *data_offset, u_long size)
__attribute__((nonnull, visibility("hidden")));
#endif /* KXLD_PIC_KEXTS */

/*******************************************************************************
* Modifiers
*******************************************************************************/

kern_return_t kxld_reloc_update_symindex(KXLDReloc *reloc, u_int symindex)
__attribute__((nonnull, visibility("hidden")));

void kxld_relocator_set_vtables(KXLDRelocator *relocator,
    const struct kxld_dict *vtables)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_relocator_process_sect_reloc(KXLDRelocator *relocator,
    const KXLDReloc *reloc, const struct kxld_sect *sect)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_relocator_process_table_reloc(KXLDRelocator *relocator,
    const KXLDReloc *reloc,
    const struct kxld_seg *seg,
    kxld_addr_t link_addr)
__attribute__((nonnull, visibility("hidden")));

#endif /* _KXLD_RELOC_H */
