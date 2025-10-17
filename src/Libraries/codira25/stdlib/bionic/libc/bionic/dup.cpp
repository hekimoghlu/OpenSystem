/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#include <unistd.h>

#include "private/bionic_fdtrack.h"

extern "C" int __dup(int old_fd);
extern "C" int __dup3(int old_fd, int new_fd, int flags);

int dup(int old_fd) {
  return FDTRACK_CREATE(__dup(old_fd));
}

int dup2(int old_fd, int new_fd) {
  // If old_fd is equal to new_fd and a valid file descriptor, dup2 returns
  // old_fd without closing it. This is not true of dup3, so we have to
  // handle this case ourselves.
  if (old_fd == new_fd) {
    if (fcntl(old_fd, F_GETFD) == -1) {
      return -1;
    }
    return old_fd;
  }

  return FDTRACK_CREATE(__dup3(old_fd, new_fd, 0));
}

int dup3(int old_fd, int new_fd, int flags) {
  return FDTRACK_CREATE(__dup3(old_fd, new_fd, flags));
}
