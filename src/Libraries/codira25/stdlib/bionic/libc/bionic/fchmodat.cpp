/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>

#include "private/ScopedFd.h"

extern "C" int __fchmodat(int, const char*, mode_t);

int fchmodat(int dirfd, const char* pathname, mode_t mode, int flags) {
  if ((flags & ~AT_SYMLINK_NOFOLLOW) != 0) {
    errno = EINVAL;
    return -1;
  }

  if (flags & AT_SYMLINK_NOFOLLOW) {
    // Emulate AT_SYMLINK_NOFOLLOW using the mechanism described
    // at https://sourceware.org/bugzilla/show_bug.cgi?id=14578
    // comment #10

    ScopedFd fd(openat(dirfd, pathname, O_PATH | O_NOFOLLOW | O_CLOEXEC));
    if (fd.get() == -1) return -1;

    // POSIX requires that ENOTSUP be returned when the system
    // doesn't support setting the mode of a symbolic link.
    // This is true for all Linux kernels.
    // We rely on the O_PATH compatibility layer added in the
    // fchmod() function to get errno correct.
    return fchmod(fd.get(), mode);
  }

  return __fchmodat(dirfd, pathname, mode);
}
