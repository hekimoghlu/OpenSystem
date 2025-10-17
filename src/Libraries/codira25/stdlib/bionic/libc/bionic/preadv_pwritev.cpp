/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#include <sys/uio.h>

// System calls we need.
extern "C" int __preadv64(int, const struct iovec*, int, long, long);
extern "C" int __preadv64v2(int, const struct iovec*, int, long, long, int);
extern "C" int __pwritev64(int, const struct iovec*, int, long, long);
extern "C" int __pwritev64v2(int, const struct iovec*, int, long, long, int);

// There is no 32-bit off_t preadv/pwritev (even on LP32).
// To avoid 32-bit ABI issues about which register pairs you're allowed
// to pass 64-bit values in, the kernel just takes two `long` arguments --
// which are int32_t for LP32, remember -- and stitches them together.
// It even does this for LP64, taking a second unused always-zero `long`.
// (The first long was int64_t, which is the same as off64_t.)
// The pair is split lo-hi (not hi-lo, as llseek is).

ssize_t preadv(int fd, const struct iovec* ios, int count, off_t offset) {
  return preadv64(fd, ios, count, offset);
}

ssize_t preadv64(int fd, const struct iovec* ios, int count, off64_t offset) {
#if defined(__LP64__)
  return __preadv64(fd, ios, count, offset, 0);
#else
  return __preadv64(fd, ios, count, offset, offset >> 32);
#endif
}

ssize_t pwritev(int fd, const struct iovec* ios, int count, off_t offset) {
  return pwritev64(fd, ios, count, offset);
}

ssize_t pwritev64(int fd, const struct iovec* ios, int count, off64_t offset) {
#if defined(__LP64__)
  return __pwritev64(fd, ios, count, offset, 0);
#else
  return __pwritev64(fd, ios, count, offset, offset >> 32);
#endif
}

ssize_t preadv2(int fd, const struct iovec* ios, int count, off_t offset, int flags) {
  return preadv64v2(fd, ios, count, offset, flags);
}

ssize_t preadv64v2(int fd, const struct iovec* ios, int count, off64_t offset, int flags) {
#if defined(__LP64__)
  return __preadv64v2(fd, ios, count, offset, 0, flags);
#else
  return __preadv64v2(fd, ios, count, offset, offset >> 32, flags);
#endif
}

ssize_t pwritev2(int fd, const struct iovec* ios, int count, off_t offset, int flags) {
  return pwritev64v2(fd, ios, count, offset, flags);
}

ssize_t pwritev64v2(int fd, const struct iovec* ios, int count, off64_t offset, int flags) {
#if defined(__LP64__)
  return __pwritev64v2(fd, ios, count, offset, 0, flags);
#else
  return __pwritev64v2(fd, ios, count, offset, offset >> 32, flags);
#endif
}
