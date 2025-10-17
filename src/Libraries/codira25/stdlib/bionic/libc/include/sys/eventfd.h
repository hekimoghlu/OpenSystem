/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
 * @file sys/eventfd.h
 * @brief Event notification file descriptors.
 */

#include <sys/cdefs.h>
#include <fcntl.h>
#include <linux/eventfd.h>

__BEGIN_DECLS

/*! \macro EFD_SEMAPHORE
 * The eventfd() flag to provide semaphore-like semantics for reads.
 */
/*! \macro EFD_CLOEXEC
 * The eventfd() flag for a close-on-exec file descriptor.
 */
/*! \macro EFD_NONBLOCK
 * The eventfd() flag for a non-blocking file descriptor.
 */

/**
 * [eventfd(2)](https://man7.org/linux/man-pages/man2/eventfd.2.html) creates a file descriptor
 * for event notification.
 *
 * Returns a new file descriptor on success, and returns -1 and sets `errno` on failure.
 */
int eventfd(unsigned int __initial_value, int __flags);

/** The type used by eventfd_read() and eventfd_write(). */
typedef uint64_t eventfd_t;

/**
 * [eventfd_read(3)](https://man7.org/linux/man-pages/man2/eventfd.2.html) is a convenience
 * wrapper to read an `eventfd_t` from an eventfd file descriptor.
 *
 * Returns 0 on success, or returns -1 otherwise.
 */
int eventfd_read(int __fd, eventfd_t* _Nonnull __value);

/**
 * [eventfd_write(3)](https://man7.org/linux/man-pages/man2/eventfd.2.html) is a convenience
 * wrapper to write an `eventfd_t` to an eventfd file descriptor.
 *
 * Returns 0 on success, or returns -1 otherwise.
 */
int eventfd_write(int __fd, eventfd_t __value);

__END_DECLS
