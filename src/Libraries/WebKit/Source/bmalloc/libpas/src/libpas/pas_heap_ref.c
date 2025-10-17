/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_heap_ref.h"

#include "pas_config.h"
#include "pas_heap.h"
#include "pas_heap_lock.h"
#include "pas_utils.h"

pas_heap* pas_ensure_heap_slow(pas_heap_ref* heap_ref,
                               pas_heap_ref_kind heap_ref_kind,
                               const pas_heap_config* config,
                               pas_heap_runtime_config* runtime_config)
{
    pas_heap* heap;

    PAS_ASSERT(heap_ref_kind != pas_fake_heap_ref_kind);

    pas_heap_lock_lock();
    heap = heap_ref->heap;
    if (!heap) {
        heap = pas_heap_create(heap_ref, heap_ref_kind, config, runtime_config);
        pas_store_store_fence();
        heap_ref->heap = heap;
    }
    pas_heap_lock_unlock();
    return heap;
}

#endif /* LIBPAS_ENABLED */
