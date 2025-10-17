/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#ifndef PAS_ALLOCATION_CONFIG_H
#define PAS_ALLOCATION_CONFIG_H

#include "pas_allocation_kind.h"

PAS_BEGIN_EXTERN_C;

struct pas_allocation_config;
typedef struct pas_allocation_config pas_allocation_config;

struct pas_allocation_config {
    void* (*allocate)(size_t size, const char* name, pas_allocation_kind allocation_kind, void* arg);
    void (*deallocate)(void* ptr, size_t size, pas_allocation_kind allocation_kind, void* arg);
    void* arg;
};

PAS_END_EXTERN_C;

#endif /* PAS_ALLOCATION_CONFIG_H */


