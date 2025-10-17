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
#ifndef _KXLD_SEG_H_
#define _KXLD_SEG_H_

#include <mach/vm_prot.h>
#include <sys/types.h>
#if KERNEL
    #include <libkern/kxld_types.h>
#else
    #include "kxld_types.h"
#endif

#include "kxld_array.h"

struct kxld_sect;
struct kxld_symtab;
struct segment_command;
struct segment_command_64;
typedef struct kxld_seg KXLDSeg;

struct kxld_seg {
	char segname[16];
	kxld_addr_t base_addr;
	kxld_addr_t link_addr;
	kxld_size_t vmsize;
	kxld_size_t fileoff;
	KXLDArray sects;
	u_int flags;
	vm_prot_t maxprot;
	vm_prot_t initprot;
};

/*******************************************************************************
* Constructors and Destructors
*******************************************************************************/

#if KXLD_USER_OR_ILP32
kern_return_t kxld_seg_init_from_macho_32(KXLDSeg *seg, struct segment_command *src)
__attribute__((nonnull, visibility("hidden")));
#endif /* KXLD_USER_OR_ILP32 */

#if KXLD_USER_OR_LP64
kern_return_t kxld_seg_init_from_macho_64(KXLDSeg *seg, struct segment_command_64 *src)
__attribute__((nonnull, visibility("hidden")));
#endif /* KXLD_USER_OR_LP64 */

#if KXLD_USER_OR_OBJECT
kern_return_t kxld_seg_create_seg_from_sections(KXLDArray *segarray,
    KXLDArray *sectarray)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_seg_finalize_object_segment(KXLDArray *segarray,
    KXLDArray *section_order, u_long hdrsize)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_seg_init_linkedit(KXLDArray *segs)
__attribute__((nonnull, visibility("hidden")));
#endif /* KXLD_USER_OR_OBJECT */

void kxld_seg_clear(KXLDSeg *seg)
__attribute__((nonnull, visibility("hidden")));

void kxld_seg_deinit(KXLDSeg *seg)
__attribute__((nonnull, visibility("hidden")));


/*******************************************************************************
* Accessors
*******************************************************************************/

kxld_size_t kxld_seg_get_vmsize(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

u_long kxld_seg_get_macho_header_size(const KXLDSeg *seg, boolean_t is_32_bit)
__attribute__((pure, nonnull, visibility("hidden")));

#if 0
/* This is no longer used, but may be useful some day... */
u_long kxld_seg_get_macho_data_size(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));
#endif

kern_return_t
kxld_seg_export_macho_to_file_buffer(const KXLDSeg *seg, u_char *buf,
    u_long *header_offset, u_long header_size,
    u_long *data_offset, u_long data_size,
    boolean_t is_32_bit)
__attribute__((nonnull, visibility("hidden")));

kern_return_t
kxld_seg_export_macho_to_vm(const KXLDSeg *seg,
    u_char *buf,
    u_long *header_offset,
    u_long header_size,
    u_long data_size,
    kxld_addr_t file_link_addr,
    boolean_t is_32_bit)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Modifiers
*******************************************************************************/

kern_return_t kxld_seg_add_section(KXLDSeg *seg, struct kxld_sect *sect)
__attribute__((nonnull, visibility("hidden")));

/* To be called after all sections are added */
kern_return_t kxld_seg_finish_init(KXLDSeg *seg)
__attribute__((nonnull, visibility("hidden")));

void kxld_seg_set_vm_protections(KXLDSeg *seg, boolean_t strict_protections)
__attribute__((nonnull, visibility("hidden")));

void kxld_seg_relocate(KXLDSeg *seg, kxld_addr_t link_addr)
__attribute__((nonnull, visibility("hidden")));

void kxld_seg_populate_linkedit(KXLDSeg *seg, const struct kxld_symtab *symtab,
    boolean_t is_32_bit
#if KXLD_PIC_KEXTS
    , const struct kxld_array *locrelocs
    , const struct kxld_array *extrelocs
    , boolean_t target_supports_slideable_kexts
#endif  /* KXLD_PIC_KEXTS */
    , uint32_t splitinfolc_size
    )
__attribute__((nonnull, visibility("hidden")));

boolean_t kxld_seg_is_split_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_seg_is_text_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_seg_is_text_exec_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_seg_is_data_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_seg_is_data_const_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_seg_is_linkedit_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

boolean_t kxld_seg_is_llvm_cov_seg(const KXLDSeg *seg)
__attribute__((pure, nonnull, visibility("hidden")));

#endif /* _KXLD_SEG_H_ */
