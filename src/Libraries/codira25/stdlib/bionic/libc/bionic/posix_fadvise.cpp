/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#include <fcntl.h>

#include "private/ErrnoRestorer.h"

extern "C" int __arm_fadvise64_64(int, int, off64_t, off64_t);
extern "C" int __fadvise64(int, off64_t, off64_t, int);

// No architecture actually has the 32-bit off_t system call.
int posix_fadvise(int fd, off_t offset, off_t length, int advice) {
  return posix_fadvise64(fd, offset, length, advice);
}

#if defined(__arm__)
int posix_fadvise64(int fd, off64_t offset, off64_t length, int advice) {
  ErrnoRestorer errno_restorer;
  return (__arm_fadvise64_64(fd, advice, offset, length) == 0) ? 0 : errno;
}
#else
int posix_fadvise64(int fd, off64_t offset, off64_t length, int advice) {
  ErrnoRestorer errno_restorer;
  return (__fadvise64(fd, offset, length, advice) == 0) ? 0 : errno;
}
#endif
