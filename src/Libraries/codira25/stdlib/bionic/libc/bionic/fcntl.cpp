/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#include <stdarg.h>
#include <fcntl.h>

#include "private/bionic_fdtrack.h"
#include "private/bionic_fortify.h"

extern "C" int __fcntl(int fd, int cmd, ...);
extern "C" int __fcntl64(int, int, ...);

int fcntl(int fd, int cmd, ...) {
  va_list args;
  va_start(args, cmd);
  // This is a bit sketchy for LP64, especially because arg can be an int,
  // but all of our supported 64-bit ABIs pass the argument in a register.
  void* arg = va_arg(args, void*);
  va_end(args);

  if (cmd == F_SETFD && (reinterpret_cast<uintptr_t>(arg) & ~FD_CLOEXEC) != 0) {
    __fortify_fatal("fcntl(F_SETFD) only supports FD_CLOEXEC but was passed %p", arg);
  }

#if defined(__LP64__)
  int rc = __fcntl(fd, cmd, arg);
#else
  // For LP32 we use the fcntl64 system call to signal that we're using struct flock64.
  int rc = __fcntl64(fd, cmd, arg);
#endif
  if (cmd == F_DUPFD) {
    return FDTRACK_CREATE_NAME("F_DUPFD", rc);
  } else if (cmd == F_DUPFD_CLOEXEC) {
    return FDTRACK_CREATE_NAME("F_DUPFD_CLOEXEC", rc);
  }
  return rc;
}
