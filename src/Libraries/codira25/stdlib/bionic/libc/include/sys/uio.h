/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
 * @file sys/uio.h
 * @brief Multi-buffer ("vector") I/O operations using `struct iovec`.
 */

#include <sys/cdefs.h>
#include <sys/types.h>
#include <linux/uio.h>

__BEGIN_DECLS

/**
 * [readv(2)](https://man7.org/linux/man-pages/man2/readv.2.html) reads
 * from an fd into the `__count` buffers described by `__iov`.
 *
 * Returns the number of bytes read on success,
 * and returns -1 and sets `errno` on failure.
 */
ssize_t readv(int __fd, const struct iovec* _Nonnull __iov, int __count);

/**
 * [writev(2)](https://man7.org/linux/man-pages/man2/writev.2.html) writes
 * to an fd from the `__count` buffers described by `__iov`.
 *
 * Returns the number of bytes written on success,
 * and returns -1 and sets `errno` on failure.
 */
ssize_t writev(int __fd, const struct iovec* _Nonnull __iov, int __count);

#if defined(__USE_GNU)

/**
 * [preadv(2)](https://man7.org/linux/man-pages/man2/preadv.2.html) reads
 * from an fd into the `__count` buffers described by `__iov`, starting at
 * offset `__offset` into the file.
 *
 * Returns the number of bytes read on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 24.
 */

#if __BIONIC_AVAILABILITY_GUARD(24)
ssize_t preadv(int __fd, const struct iovec* _Nonnull __iov, int __count, off_t __offset) __RENAME_IF_FILE_OFFSET64(preadv64) __INTRODUCED_IN(24);

/**
 * [pwritev(2)](https://man7.org/linux/man-pages/man2/pwritev.2.html) writes
 * to an fd from the `__count` buffers described by `__iov`, starting at offset
 * `__offset` into the file.
 *
 * Returns the number of bytes written on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 24.
 */
ssize_t pwritev(int __fd, const struct iovec* _Nonnull __iov, int __count, off_t __offset) __RENAME_IF_FILE_OFFSET64(pwritev64) __INTRODUCED_IN(24);

/**
 * Like preadv() but with a 64-bit offset even in a 32-bit process.
 *
 * Available since API level 24.
 */
ssize_t preadv64(int __fd, const struct iovec* _Nonnull __iov, int __count, off64_t __offset) __INTRODUCED_IN(24);

/**
 * Like pwritev() but with a 64-bit offset even in a 32-bit process.
 *
 * Available since API level 24.
 */
ssize_t pwritev64(int __fd, const struct iovec* _Nonnull __iov, int __count, off64_t __offset) __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */


/**
 * [preadv2(2)](https://man7.org/linux/man-pages/man2/preadv2.2.html) reads
 * from an fd into the `__count` buffers described by `__iov`, starting at
 * offset `__offset` into the file, with the given flags.
 *
 * Returns the number of bytes read on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 33.
 */

#if __BIONIC_AVAILABILITY_GUARD(33)
ssize_t preadv2(int __fd, const struct iovec* _Nonnull __iov, int __count, off_t __offset, int __flags) __RENAME_IF_FILE_OFFSET64(preadv64v2) __INTRODUCED_IN(33);

/**
 * [pwritev2(2)](https://man7.org/linux/man-pages/man2/pwritev2.2.html) writes
 * to an fd from the `__count` buffers described by `__iov`, starting at offset
 * `__offset` into the file, with the given flags.
 *
 * Returns the number of bytes written on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 33.
 */
ssize_t pwritev2(int __fd, const struct iovec* _Nonnull __iov, int __count, off_t __offset, int __flags) __RENAME_IF_FILE_OFFSET64(pwritev64v2) __INTRODUCED_IN(33);

/**
 * Like preadv2() but with a 64-bit offset even in a 32-bit process.
 *
 * Available since API level 33.
 */
ssize_t preadv64v2(int __fd, const struct iovec* _Nonnull __iov, int __count, off64_t __offset, int __flags) __INTRODUCED_IN(33);

/**
 * Like pwritev2() but with a 64-bit offset even in a 32-bit process.
 *
 * Available since API level 33.
 */
ssize_t pwritev64v2(int __fd, const struct iovec* _Nonnull __iov, int __count, off64_t __offset, int __flags) __INTRODUCED_IN(33);
#endif /* __BIONIC_AVAILABILITY_GUARD(33) */


/**
 * [process_vm_readv(2)](https://man7.org/linux/man-pages/man2/process_vm_readv.2.html)
 * reads from the address space of another process.
 *
 * Returns the number of bytes read on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
ssize_t process_vm_readv(pid_t __pid, const struct iovec* __BIONIC_COMPLICATED_NULLNESS __local_iov, unsigned long __local_iov_count, const struct iovec* __BIONIC_COMPLICATED_NULLNESS __remote_iov, unsigned long __remote_iov_count, unsigned long __flags) __INTRODUCED_IN(23);

/**
 * [process_vm_writev(2)](https://man7.org/linux/man-pages/man2/process_vm_writev.2.html)
 * writes to the address space of another process.
 *
 * Returns the number of bytes read on success,
 * and returns -1 and sets `errno` on failure.
 *
 * Available since API level 23.
 */
ssize_t process_vm_writev(pid_t __pid, const struct iovec* __BIONIC_COMPLICATED_NULLNESS __local_iov, unsigned long __local_iov_count, const struct iovec* __BIONIC_COMPLICATED_NULLNESS __remote_iov, unsigned long __remote_iov_count, unsigned long __flags) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


#endif

__END_DECLS
