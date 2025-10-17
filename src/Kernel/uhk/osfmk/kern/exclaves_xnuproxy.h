/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#include <mach/exclaves_l4.h>

#include "exclaves_internal.h"

#include "kern/exclaves.tightbeam.h"

__BEGIN_DECLS

extern kern_return_t
exclaves_xnuproxy_pmm_usage(void);

extern kern_return_t
exclaves_xnuproxy_ctx_alloc(exclaves_ctx_t *ctx);

extern kern_return_t
exclaves_xnuproxy_ctx_free(exclaves_ctx_t *ctx);

extern kern_return_t
exclaves_xnuproxy_init(uint64_t bootinfo_pa);

/* BEGIN IGNORE CODESTYLE */
/*
 * Note: strings passed to callback are not valid outside of the context of the
 * callback.
 */
extern kern_return_t
exclaves_xnuproxy_resource_info(void (^cb)(const char *name, const char *domain,
    xnuproxy_resourcetype_s, uint64_t id, bool));
/* END IGNORE CODESTYLE */

extern kern_return_t
exclaves_xnuproxy_audio_buffer_copyout(uint64_t id,
    uint64_t size1, uint64_t offset1, uint64_t size2, uint64_t offset2);

extern kern_return_t
exclaves_xnuproxy_audio_buffer_delete(uint64_t id);

extern kern_return_t
exclaves_xnuproxy_audio_buffer_map(uint64_t id, size_t size, bool *read_only);

/* BEGIN IGNORE CODESTYLE */
extern kern_return_t
exclaves_xnuproxy_audio_buffer_layout(uint64_t id, uint32_t start,
    uint32_t npages, kern_return_t (^cb)(uint64_t base, uint32_t npages));
/* ENDIGNORE CODESTYLE */

extern kern_return_t
exclaves_xnuproxy_named_buffer_delete(uint64_t id);

extern kern_return_t
exclaves_xnuproxy_named_buffer_map(uint64_t id, size_t size, bool *read_only);

/* BEGIN IGNORE CODESTYLE */
extern kern_return_t
exclaves_xnuproxy_named_buffer_layout(uint64_t id, uint32_t start,
    uint32_t npages, kern_return_t (^cb)(uint64_t base, uint32_t npages));
/* END IGNORE CODESTYLE */

extern kern_return_t
exclaves_xnuproxy_endpoint_call(Exclaves_L4_Word_t endpoint_id);

__END_DECLS

#endif /* CONFIG_EXCLAVES */
