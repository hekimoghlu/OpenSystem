/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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

#if __riscv

/**
 * @file sys/hwprobe.h
 * @brief RISC-V hardware probing.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

/* Pull in struct riscv_hwprobe and corresponding constants. */
#include <asm/hwprobe.h>

__BEGIN_DECLS

/**
 * [__riscv_hwprobe(2)](https://docs.kernel.org/riscv/hwprobe.html)
 * queries hardware characteristics.
 *
 * A `__cpu_count` of 0 and null `__cpus` means "all online cpus".
 *
 * Returns 0 on success and returns an error number on failure.
 */
int __riscv_hwprobe(struct riscv_hwprobe* _Nonnull __pairs, size_t __pair_count, size_t __cpu_count, unsigned long* _Nullable __cpus, unsigned __flags);

/**
 * The type of the second argument passed to riscv64 ifunc resolvers.
 * This argument allows riscv64 ifunc resolvers to call __riscv_hwprobe()
 * without worrying about whether that relocation is resolved before
 * the ifunc resolver is called.
 */
typedef int (*__riscv_hwprobe_t)(struct riscv_hwprobe* _Nonnull __pairs, size_t __pair_count, size_t __cpu_count, unsigned long* _Nullable __cpus, unsigned __flags);

__END_DECLS

#endif
