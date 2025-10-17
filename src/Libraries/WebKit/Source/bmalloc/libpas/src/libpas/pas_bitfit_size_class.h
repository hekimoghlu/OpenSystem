/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#ifndef PAS_BITFIT_SIZE_CLASS_H
#define PAS_BITFIT_SIZE_CLASS_H

#include "pas_compact_atomic_bitfit_size_class_ptr.h"
#include "pas_compact_bitfit_directory_ptr.h"
#include "pas_utils.h"
#include "pas_versioned_field.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_size_class;
struct pas_bitfit_page_config;
struct pas_bitfit_size_class;
struct pas_bitfit_view;
typedef struct pas_bitfit_size_class pas_bitfit_size_class;
typedef struct pas_bitfit_page_config pas_bitfit_page_config;
typedef struct pas_bitfit_size_class pas_bitfit_size_class;
typedef struct pas_bitfit_view pas_bitfit_view;

struct PAS_ALIGNED(sizeof(pas_versioned_field)) pas_bitfit_size_class {
    pas_versioned_field first_free;
    unsigned size;
    pas_compact_atomic_bitfit_size_class_ptr next_smaller;
    pas_compact_bitfit_directory_ptr directory;
};

PAS_API pas_compact_atomic_bitfit_size_class_ptr*
pas_bitfit_size_class_find_insertion_point(pas_bitfit_directory* directory,
                                           unsigned size);

PAS_API void pas_bitfit_size_class_construct(
    pas_bitfit_size_class* size_class,
    unsigned size,
    pas_bitfit_directory* directory,
    pas_compact_atomic_bitfit_size_class_ptr* insertion_point);

PAS_API pas_bitfit_view*
pas_bitfit_size_class_get_first_free_view(pas_bitfit_size_class* size_class,
                                          const pas_bitfit_page_config* page_config);

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_SIZE_CLASS_H */

