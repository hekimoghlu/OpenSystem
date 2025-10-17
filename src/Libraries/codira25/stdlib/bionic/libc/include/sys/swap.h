/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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

/**
 * @file sys/swap.h
 * @brief Swap control.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/** swapon() flag to discard pages. */
#define SWAP_FLAG_DISCARD 0x10000

/**
 * swapon() flag to give this swap area a non-default priority.
 * The priority is also encoded in the flags:
 * `(priority << SWAP_FLAG_PRIO_SHIFT) & SWAP_FLAG_PRIO_MASK`.
 */
#define SWAP_FLAG_PREFER 0x8000
/** See SWAP_FLAG_PREFER. */
#define SWAP_FLAG_PRIO_MASK 0x7fff
/** See SWAP_FLAG_PREFER. */
#define SWAP_FLAG_PRIO_SHIFT 0

/**
 * [swapon(2)](https://man7.org/linux/man-pages/man2/swapon.2.html) enables swapping.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int swapon(const char* _Nonnull __path,  int __flags);

/**
 * [swapoff(2)](https://man7.org/linux/man-pages/man2/swapoff.2.html) disables swapping.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 */
int swapoff(const char* _Nonnull __path);

__END_DECLS
