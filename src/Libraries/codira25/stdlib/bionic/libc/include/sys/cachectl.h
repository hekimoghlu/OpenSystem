/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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

/*
 * @file sys/cachectl.h
 * @brief Architecture-specific cache control.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

#if defined(__riscv)

/**
 * Flag for __riscv_flush_icache() to indicate that only the current
 * thread's instruction cache needs to be flushed (rather than the
 * default of all threads).
 */
#define SYS_RISCV_FLUSH_ICACHE_LOCAL 1UL

/**
 * __riscv_flush_icache(2) flushes the instruction cache for the given range of addresses.
 * The address range is currently (Linux 6.12) ignored, so both pointers may be null.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int __riscv_flush_icache(void* _Nullable __start, void* _Nullable __end, unsigned long __flags);

#endif

__END_DECLS
