/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

#include <stdatomic.h>
#include <stdio.h>

#include <async_safe/log.h>
#include <private/bionic_globals.h>
#include <private/bionic_malloc_dispatch.h>

#if __has_feature(hwaddress_sanitizer)

#include <sanitizer/hwasan_interface.h>

__BEGIN_DECLS

// FIXME: implement these in HWASan allocator.
int __sanitizer_malloc_iterate(uintptr_t base, size_t size,
                               void (*callback)(uintptr_t base, size_t size, void* arg),
                               void* arg);
void __sanitizer_malloc_disable();
void __sanitizer_malloc_enable();
int __sanitizer_malloc_info(int options, FILE* fp);

__END_DECLS

#define Malloc(function)  __sanitizer_ ## function

#else // __has_feature(hwaddress_sanitizer)

#ifdef  __LP64__
#ifndef USE_H_MALLOC
#error missing USE_H_MALLOC
#endif

#include "h_malloc.h"
#define Malloc(function)  h_ ## function
__BEGIN_DECLS
int h_malloc_info(int options, FILE* fp);
__END_DECLS

#if defined(USE_SCUDO)
#include "scudo.h"
void InitNativeAllocatorDispatch(libc_globals* globals);
#endif

#define BOTH_H_MALLOC_AND_SCUDO

#else // 32-bit
#include "scudo.h"
#define Malloc(function)  scudo_ ## function
#endif

#endif

const MallocDispatch* NativeAllocatorDispatch();

static inline const MallocDispatch* GetDispatchTable() {
  return atomic_load_explicit(&__libc_globals->current_dispatch_table, memory_order_acquire);
}

static inline const MallocDispatch* GetDefaultDispatchTable() {
  return atomic_load_explicit(&__libc_globals->default_dispatch_table, memory_order_acquire);
}

// =============================================================================
// Log functions
// =============================================================================
#define error_log(format, ...)  \
    async_safe_format_log(ANDROID_LOG_ERROR, "libc", (format), ##__VA_ARGS__ )
#define info_log(format, ...)  \
    async_safe_format_log(ANDROID_LOG_INFO, "libc", (format), ##__VA_ARGS__ )
#define warning_log(format, ...)  \
    async_safe_format_log(ANDROID_LOG_WARN, "libc", (format), ##__VA_ARGS__ )
// =============================================================================
