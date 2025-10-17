/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#include <unistd.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>

int lockf64(int fd, int cmd, off64_t length) {
  // Translate POSIX lockf into fcntl.
  struct flock64 fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_whence = SEEK_CUR;
  fl.l_start = 0;
  fl.l_len = length;

  if (cmd == F_ULOCK) {
    fl.l_type = F_UNLCK;
    cmd = F_SETLK64;
    return fcntl(fd, F_SETLK64, &fl);
  }

  if (cmd == F_LOCK) {
    fl.l_type = F_WRLCK;
    return fcntl(fd, F_SETLKW64, &fl);
  }

  if (cmd == F_TLOCK) {
    fl.l_type = F_WRLCK;
    return fcntl(fd, F_SETLK64, &fl);
  }

  if (cmd == F_TEST) {
    fl.l_type = F_RDLCK;
    if (fcntl(fd, F_GETLK64, &fl) == -1) return -1;
    if (fl.l_type == F_UNLCK || fl.l_pid == getpid()) return 0;
    errno = EACCES;
    return -1;
  }

  errno = EINVAL;
  return -1;
}

#if defined(__LP64__)
// For LP64, off_t == off64_t.
__strong_alias(lockf, lockf64);
#else
// For ILP32 we need a shim that truncates the off64_t to off_t.
int lockf(int fd, int cmd, off_t length) {
  return lockf64(fd, cmd, length);
}
#endif
