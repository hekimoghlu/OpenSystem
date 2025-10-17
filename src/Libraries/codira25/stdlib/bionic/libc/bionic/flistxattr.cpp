/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>

#include "private/FdPath.h"

extern "C" ssize_t __flistxattr(int, char*, size_t);

ssize_t flistxattr(int fd, char* list, size_t size) {
  int saved_errno = errno;
  ssize_t result = __flistxattr(fd, list, size);
  if (result != -1 || errno != EBADF) {
    return result;
  }

  // fd could be an O_PATH file descriptor, and the kernel
  // may not directly support flistxattr() on such a file descriptor.
  // Use /proc/self/fd instead to emulate this support.
  int fd_flag = fcntl(fd, F_GETFL);
  if (fd_flag == -1 || (fd_flag & O_PATH) == 0) {
    errno = EBADF;
    return -1;
  }

  errno = saved_errno;
  return listxattr(FdPath(fd).c_str(), list, size);
}
