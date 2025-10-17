/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef PAS_BOOTSTRAP_HEAP_PAGE_PROVIDER_H
#define PAS_BOOTSTRAP_HEAP_PAGE_PROVIDER_H

#include "pas_heap_page_provider.h"

PAS_BEGIN_EXTERN_C;

PAS_API pas_allocation_result pas_bootstrap_heap_page_provider(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_heap* heap,
    pas_physical_memory_transaction* transaction,
    void *arg);

PAS_END_EXTERN_C;

#endif /* PAS_BOOTSTRAP_HEAP_PAGE_PROVIDER_H */

