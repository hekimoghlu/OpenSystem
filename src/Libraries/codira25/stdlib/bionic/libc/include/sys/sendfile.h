/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
 * @file sys/sendfile.h
 * @brief The sendfile() function.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

/* See https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md */
#if defined(__USE_FILE_OFFSET64)
ssize_t sendfile(int __out_fd, int __in_fd, off_t* _Nullable __offset, size_t __count) __RENAME(sendfile64);
#else
/**
 * [sendfile(2)](https://man7.org/linux/man-pages/man2/sendfile.2.html) copies data directly
 * between two file descriptors.
 *
 * Returns the number of bytes copied on success, and returns -1 and sets `errno` on failure.
 */
ssize_t sendfile(int __out_fd, int __in_fd, off_t* _Nullable __offset, size_t __count);
#endif

/**
 * Like sendfile() but allows using a 64-bit offset
 * even from a 32-bit process without `_FILE_OFFSET_BITS=64`.
 */
ssize_t sendfile64(int __out_fd, int __in_fd, off64_t* _Nullable __offset, size_t __count);

__END_DECLS
