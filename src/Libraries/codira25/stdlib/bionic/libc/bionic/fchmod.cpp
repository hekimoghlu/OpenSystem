/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include <stdio.h>

#include "private/FdPath.h"

extern "C" int __fchmod(int, mode_t);

int fchmod(int fd, mode_t mode) {
  int saved_errno = errno;
  int result = __fchmod(fd, mode);
  if (result == 0 || errno != EBADF) {
    return result;
  }

  // fd could be an O_PATH file descriptor, and the kernel
  // may not directly support fchmod() on such a file descriptor.
  // Use /proc/self/fd instead to emulate this support.
  // https://sourceware.org/bugzilla/show_bug.cgi?id=14578
  //
  // As of February 2015, there are no kernels which support fchmod
  // on an O_PATH file descriptor, and "man open" documents fchmod
  // on O_PATH file descriptors as returning EBADF.
  int fd_flag = fcntl(fd, F_GETFL);
  if (fd_flag == -1 || (fd_flag & O_PATH) == 0) {
    errno = EBADF;
    return -1;
  }

  errno = saved_errno;
  result = chmod(FdPath(fd).c_str(), mode);
  if (result == -1 && errno == ELOOP) {
    // Linux does not support changing the mode of a symlink.
    // For fchmodat(AT_SYMLINK_NOFOLLOW), POSIX requires a return
    // value of ENOTSUP. Assume that's true here too.
    errno = ENOTSUP;
  }

  return result;
}
