/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#if CONFIG_EXCLAVES

#pragma once

#include <stdint.h>
#include <mach/kern_return.h>

#include "kern/exclaves.tightbeam.h"

/* The maximum number of pages in a memory request. */
#define EXCLAVES_MEMORY_MAX_REQUEST (64)

typedef enum : uint32_t {
	EXCLAVES_MEMORY_PAGEKIND_ROOTDOMAIN = 1,
	EXCLAVES_MEMORY_PAGEKIND_CONCLAVE = 2,
} exclaves_memory_pagekind_t;

typedef enum __enum_closed __enum_options : uint32_t {
	EXCLAVES_MEMORY_PAGE_FLAGS_NONE = 0,
} exclaves_memory_page_flags_t;

__BEGIN_DECLS

extern void
exclaves_memory_alloc(uint32_t npages, uint32_t * _Nonnull pages, const exclaves_memory_pagekind_t kind, const exclaves_memory_page_flags_t flags);

extern void
exclaves_memory_free(uint32_t npages, const uint32_t * _Nonnull pages, const exclaves_memory_pagekind_t kind, const exclaves_memory_page_flags_t flags);

extern kern_return_t
exclaves_memory_map(uint32_t npages, const uint32_t * _Nonnull pages, vm_prot_t prot,
    char * _Nullable * _Nonnull address);

extern kern_return_t
exclaves_memory_unmap(char * _Nonnull address, size_t size);

/* BEGIN IGNORE CODESTYLE */

/* Legacy upcall handlers */

extern tb_error_t
exclaves_memory_upcall_legacy_alloc(uint32_t npages, xnuupcalls_pagekind_s kind,
    tb_error_t (^_Nonnull completion)(xnuupcalls_pagelist_s));

extern tb_error_t
exclaves_memory_upcall_legacy_alloc_ext(uint32_t npages, xnuupcalls_pageallocflags_s flags,
    tb_error_t (^_Nonnull completion)(xnuupcalls_pagelist_s));

extern tb_error_t
exclaves_memory_upcall_legacy_free(const uint32_t pages[_Nonnull EXCLAVES_MEMORY_MAX_REQUEST],
    uint32_t npages, const xnuupcalls_pagekind_s kind,
    tb_error_t (^_Nonnull completion)(void));

extern tb_error_t
exclaves_memory_upcall_legacy_free_ext(const uint32_t pages[_Nonnull EXCLAVES_MEMORY_MAX_REQUEST],
    uint32_t npages, xnuupcalls_pagefreeflags_s flags,
    tb_error_t (^_Nonnull completion)(void));

/* Upcall handlers */

extern tb_error_t
exclaves_memory_upcall_alloc(uint32_t npages, xnuupcallsv2_pagekind_s kind,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_pagelist_s));

extern tb_error_t
exclaves_memory_upcall_alloc_ext(uint32_t npages, xnuupcallsv2_pageallocflagsv2_s flags,
    tb_error_t (^_Nonnull completion)(xnuupcallsv2_pagelist_s));

extern tb_error_t
exclaves_memory_upcall_free(const xnuupcallsv2_pagelist_s pages,
    const xnuupcallsv2_pagekind_s kind, tb_error_t (^_Nonnull completion)(void));

extern tb_error_t
exclaves_memory_upcall_free_ext(const xnuupcallsv2_pagelist_s pages,
    const xnuupcallsv2_pagefreeflagsv2_s kind, tb_error_t (^_Nonnull completion)(void));

/* END IGNORE CODESTYLE */

extern void
exclaves_memory_report_accounting(void);

__END_DECLS

#endif /* CONFIG_EXCLAVES */
