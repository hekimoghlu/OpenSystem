/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#pragma once

#include <bionic/pthread_internal.h>
#include <platform/bionic/malloc.h>
#include <stddef.h>

// Expected to be called in a single-threaded context during libc init, so no
// synchronization required.
void SetDefaultHeapTaggingLevel();

// Lock for the heap tagging level. You may find ScopedPthreadMutexLocker
// useful for RAII on this lock.
extern pthread_mutex_t g_heap_tagging_lock;

bool BlockHeapTaggingLevelDowngrade();

// This function can be called in a multithreaded context, and thus should
// only be called when holding the `g_heap_tagging_lock`.
bool SetHeapTaggingLevel(HeapTaggingLevel level);

// This is static because libc_nomalloc uses this but does not need to link the
// cpp file.
__attribute__((unused)) static inline const char* DescribeTaggingLevel(
    HeapTaggingLevel level) {
  switch (level) {
    case M_HEAP_TAGGING_LEVEL_NONE:
      return "none";
    case M_HEAP_TAGGING_LEVEL_TBI:
      return "tbi";
    case M_HEAP_TAGGING_LEVEL_ASYNC:
      return "async";
    case M_HEAP_TAGGING_LEVEL_SYNC:
      return "sync";
  }
}
