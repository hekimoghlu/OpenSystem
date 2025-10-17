/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#ifndef ProcessScavenger_h
#define ProcessScavenger_h

#include <stdint.h>
#include <stdbool.h>
#include <mach/mach_types.h>

#include "Defines.h"

// Allocator cannot be imported into Swift currently, so expose C entry points here. These functions allocate the buffer and it
// is the callers responsibility to free it
#if __cplusplus
extern "C" {
#endif
VIS_HIDDEN bool scavengeProcess(task_read_t task, void** buffer, uint64_t* bufferSize);
VIS_HIDDEN void* scavengeCache(const char* path, uint64_t* bufferSize);
#if __cplusplus
}
#endif
#endif /* ProcessScavenger_h */
