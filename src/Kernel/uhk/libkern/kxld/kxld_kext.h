/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#ifndef _KXLD_KEXT_H_
#define _KXLD_KEXT_H_

#include <sys/types.h>
#if KERNEL
    #include <libkern/kxld_types.h>
#else
    #include "kxld_types.h"
#endif

struct kxld_array;
struct kxld_kext;
struct kxld_dict;
struct kxld_object;
struct kxld_sect;
struct kxld_seg;
struct kxld_symtab;
struct kxld_vtable;
typedef struct kxld_kext KXLDKext;

/*******************************************************************************
* Constructors and Destructors
*******************************************************************************/

size_t kxld_kext_sizeof(void)
__attribute__((const, visibility("hidden")));

kern_return_t kxld_kext_init(KXLDKext *kext, struct kxld_object *kext_object,
    struct kxld_object *interface_object)
__attribute__((nonnull(1, 2), visibility("hidden")));

void kxld_kext_clear(KXLDKext *kext)
__attribute__((nonnull, visibility("hidden")));

void kxld_kext_deinit(KXLDKext *kext)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Accessors
*******************************************************************************/

kern_return_t kxld_kext_export_symbols(const KXLDKext *kext,
    struct kxld_dict *defined_symbols_by_name,
    struct kxld_dict *obsolete_symbols_by_name,
    struct kxld_dict *defined_cxx_symbols_by_value)
__attribute__((nonnull(1), visibility("hidden")));

void kxld_kext_get_vmsize_for_seg_by_name(const KXLDKext *kext,
    const char *segname,
    u_long *vmsize)
__attribute__((nonnull, visibility("hidden")));

void kxld_kext_get_vmsize(const KXLDKext *kext,
    u_long *header_size, u_long *vmsize)
__attribute__((nonnull, visibility("hidden")));

void kxld_kext_set_linked_object_size(KXLDKext *kext, u_long vmsize)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_kext_export_linked_object(const KXLDKext *kext,
    void *linked_object,
    kxld_addr_t *kmod_info)
__attribute__((nonnull, visibility("hidden")));

/*******************************************************************************
* Modifiers
*******************************************************************************/
kern_return_t kxld_kext_export_vtables(KXLDKext *kext,
    const struct kxld_dict *defined_cxx_symbols,
    const struct kxld_dict *defined_symbols,
    struct kxld_dict *vtables)
__attribute__((nonnull, visibility("hidden")));

kern_return_t kxld_kext_relocate(KXLDKext *kext,
    kxld_addr_t link_address,
    struct kxld_dict *patched_vtables,
    const struct kxld_dict *defined_symbols,
    const struct kxld_dict *obsolete_symbols,
    const struct kxld_dict *defined_cxx_symbols)
__attribute__((nonnull(1, 3, 4), visibility("hidden")));


#endif /* _KXLD_KEXT_H_ */
