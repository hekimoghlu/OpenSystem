/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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

#ifndef MEMORY_H
#define MEMORY_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __linux__
#define HAVE_COMPATIBLE_MREMAP
#endif

int get_metadata_key(void);

void *memory_map(size_t size);
#ifdef HAS_ARM_MTE
void *memory_map_mte(size_t size);
#endif
bool memory_map_fixed(void *ptr, size_t size);
#ifdef HAS_ARM_MTE
bool memory_map_fixed_mte(void *ptr, size_t size);
#endif
bool memory_unmap(void *ptr, size_t size);
bool memory_protect_ro(void *ptr, size_t size);
bool memory_protect_rw(void *ptr, size_t size);
bool memory_protect_rw_metadata(void *ptr, size_t size);
#ifdef HAVE_COMPATIBLE_MREMAP
bool memory_remap(void *old, size_t old_size, size_t new_size);
bool memory_remap_fixed(void *old, size_t old_size, void *new, size_t new_size);
#endif
bool memory_purge(void *ptr, size_t size);
bool memory_set_name(void *ptr, size_t size, const char *name);

#endif
