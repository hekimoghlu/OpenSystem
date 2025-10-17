/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
 * @file sys/signalfd.h
 * @brief File-descriptor based signal interface.
 */

#include <sys/cdefs.h>

#include <linux/signalfd.h>
#include <signal.h>

__BEGIN_DECLS

/**
 * [signalfd(2)](https://man7.org/linux/man-pages/man2/signalfd.2.html) creates/manipulates a
 * file descriptor for reading signal events.
 *
 * Returns the file descriptor on success, and returns -1 and sets `errno` on failure.
 */
int signalfd(int __fd, const sigset_t* _Nonnull __mask, int __flags);

/**
 * Like signalfd() but allows setting a signal mask with RT signals even from a 32-bit process.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
int signalfd64(int __fd, const sigset64_t* _Nonnull __mask, int __flags) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


__END_DECLS
