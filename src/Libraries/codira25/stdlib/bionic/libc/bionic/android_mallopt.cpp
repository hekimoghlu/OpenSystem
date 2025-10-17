/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#include <errno.h>
#include <stdatomic.h>

#include <platform/bionic/malloc.h>
#include <private/bionic_globals.h>

#include "gwp_asan_wrappers.h"
#include "malloc_limit.h"

#if !defined(LIBC_STATIC)
#include <stdio.h>

#include <private/bionic_defs.h>

#include "malloc_heapprofd.h"

extern bool gZygoteChild;
extern _Atomic bool gZygoteChildProfileable;

bool WriteMallocLeakInfo(FILE* fp);
bool GetMallocLeakInfo(android_mallopt_leak_info_t* leak_info);
bool FreeMallocLeakInfo(android_mallopt_leak_info_t* leak_info);
#endif

// =============================================================================
// Platform-internal mallopt variant.
// =============================================================================
#if !defined(LIBC_STATIC)
__BIONIC_WEAK_FOR_NATIVE_BRIDGE
#endif
extern "C" bool android_mallopt(int opcode, void* arg, size_t arg_size) {
  // Functionality available in both static and dynamic libc.
  if (opcode == M_GET_DECAY_TIME_ENABLED) {
    if (arg == nullptr || arg_size != sizeof(bool)) {
      errno = EINVAL;
      return false;
    }
    *reinterpret_cast<bool*>(arg) = atomic_load(&__libc_globals->decay_time_enabled);
    return true;
  }
  if (opcode == M_MEMTAG_STACK_IS_ON) {
    if (arg == nullptr || arg_size != sizeof(bool)) {
      errno = EINVAL;
      return false;
    }
    *reinterpret_cast<bool*>(arg) = atomic_load(&__libc_memtag_stack);
    return true;
  }
  if (opcode == M_SET_ALLOCATION_LIMIT_BYTES) {
    return LimitEnable(arg, arg_size);
  }

#if defined(LIBC_STATIC)
  errno = ENOTSUP;
  return false;
#else
  if (opcode == M_SET_ZYGOTE_CHILD) {
    if (arg != nullptr || arg_size != 0) {
      errno = EINVAL;
      return false;
    }
    gZygoteChild = true;
    return true;
  }
  if (opcode == M_INIT_ZYGOTE_CHILD_PROFILING) {
    if (arg != nullptr || arg_size != 0) {
      errno = EINVAL;
      return false;
    }
    atomic_store_explicit(&gZygoteChildProfileable, true, memory_order_release);
    // Also check if heapprofd should start profiling from app startup.
    HeapprofdInitZygoteChildProfiling();
    return true;
  }
  if (opcode == M_GET_PROCESS_PROFILEABLE) {
    if (arg == nullptr || arg_size != sizeof(bool)) {
      errno = EINVAL;
      return false;
    }
    // Native processes are considered profileable. Zygote children are considered
    // profileable only when appropriately tagged.
    *reinterpret_cast<bool*>(arg) =
        !gZygoteChild || atomic_load_explicit(&gZygoteChildProfileable, memory_order_acquire);
    return true;
  }
  if (opcode == M_WRITE_MALLOC_LEAK_INFO_TO_FILE) {
    if (arg == nullptr || arg_size != sizeof(FILE*)) {
      errno = EINVAL;
      return false;
    }
    return WriteMallocLeakInfo(reinterpret_cast<FILE*>(arg));
  }
  if (opcode == M_GET_MALLOC_LEAK_INFO) {
    if (arg == nullptr || arg_size != sizeof(android_mallopt_leak_info_t)) {
      errno = EINVAL;
      return false;
    }
    return GetMallocLeakInfo(reinterpret_cast<android_mallopt_leak_info_t*>(arg));
  }
  if (opcode == M_FREE_MALLOC_LEAK_INFO) {
    if (arg == nullptr || arg_size != sizeof(android_mallopt_leak_info_t)) {
      errno = EINVAL;
      return false;
    }
    return FreeMallocLeakInfo(reinterpret_cast<android_mallopt_leak_info_t*>(arg));
  }
  // Try heapprofd's mallopt, as it handles options not covered here.
  return HeapprofdMallopt(opcode, arg, arg_size);
#endif
}
