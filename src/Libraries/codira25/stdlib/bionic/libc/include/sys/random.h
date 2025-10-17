/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
 * @file sys/random.h
 * @brief The getentropy() and getrandom() functions.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

#include <linux/random.h>

#include <bits/getentropy.h>

__BEGIN_DECLS

/**
 * [getrandom(2)](https://man7.org/linux/man-pages/man2/getrandom.2.html) fills the given buffer
 * with random bytes.
 *
 * Returns the number of bytes copied on success, and returns -1 and sets `errno` on failure.
 *
 * Available since API level 28.
 *
 * See also arc4random_buf() which is available in all API levels.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
__nodiscard ssize_t getrandom(void* _Nonnull __buffer, size_t __buffer_size, unsigned int __flags) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


__END_DECLS
