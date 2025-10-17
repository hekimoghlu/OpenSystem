/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#ifndef PAS_BITFIT_ALLOCATOR_H
#define PAS_BITFIT_ALLOCATOR_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_allocator;
struct pas_bitfit_size_class;
struct pas_bitfit_page;
struct pas_bitfit_size_class;
struct pas_bitfit_view;
typedef struct pas_bitfit_allocator pas_bitfit_allocator;
typedef struct pas_bitfit_size_class pas_bitfit_size_class;
typedef struct pas_bitfit_page pas_bitfit_page;
typedef struct pas_bitfit_size_class pas_bitfit_size_class;
typedef struct pas_bitfit_view pas_bitfit_view;

struct pas_bitfit_allocator {
    pas_bitfit_size_class* size_class;
    pas_bitfit_view* view;
};

PAS_API void pas_bitfit_allocator_construct(pas_bitfit_allocator* allocator,
                                            pas_bitfit_size_class* size_class);

PAS_API void pas_bitfit_allocator_stop(pas_bitfit_allocator* allocator);

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_ALLOCATOR_H */

