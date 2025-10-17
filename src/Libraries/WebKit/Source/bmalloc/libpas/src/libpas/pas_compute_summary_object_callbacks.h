/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#ifndef PAS_COMPUTE_SUMMARY_OBJECT_CALLBACKS_H
#define PAS_COMPUTE_SUMMARY_OBJECT_CALLBACKS_H

#include "pas_heap_config.h"
#include "pas_large_free.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

PAS_API bool pas_compute_summary_live_object_callback(
    uintptr_t begin,
    uintptr_t end,
    void* arg);

PAS_API bool pas_compute_summary_live_object_callback_without_physical_sharing(
    uintptr_t begin,
    uintptr_t end,
    void* arg);

PAS_API bool (*pas_compute_summary_live_object_callback_for_config(const pas_heap_config* config))(
    uintptr_t begin,
    uintptr_t end,
    void* arg);

PAS_API bool pas_compute_summary_dead_object_callback(
    pas_large_free free,
    void* arg);

PAS_API bool pas_compute_summary_dead_object_callback_without_physical_sharing(
    pas_large_free free,
    void* arg);

PAS_API bool (*pas_compute_summary_dead_object_callback_for_config(const pas_heap_config* config))(
    pas_large_free free,
    void* arg);

PAS_END_EXTERN_C;

#endif /* PAS_COMPUTE_SUMMARY_OBJECT_CALLBACKS_H */

