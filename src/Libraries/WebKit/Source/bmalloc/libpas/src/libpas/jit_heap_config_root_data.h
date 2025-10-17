/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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
#ifndef JIT_HEAP_CONFIG_ROOT_DATA_H
#define JIT_HEAP_CONFIG_ROOT_DATA_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct jit_heap_config_root_data;
struct pas_page_header_table;
typedef struct jit_heap_config_root_data jit_heap_config_root_data;
typedef struct pas_page_header_table pas_page_header_table;

struct jit_heap_config_root_data {
    pas_page_header_table* small_page_header_table;
    pas_page_header_table* medium_page_header_table;
};

PAS_END_EXTERN_C;

#endif /* JIT_HEAP_CONFIG_ROOT_DATA_H */

