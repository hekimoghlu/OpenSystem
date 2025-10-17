/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#include <sys/time.h>

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "private/bionic_time_conversions.h"

static int futimesat(int fd, const char* path, const timeval tv[2], int flags) {
  timespec ts[2];
  if (tv && (!timespec_from_timeval(ts[0], tv[0]) || !timespec_from_timeval(ts[1], tv[1]))) {
    errno = EINVAL;
    return -1;
  }
  return utimensat(fd, path, tv ? ts : nullptr, flags);
}

int utimes(const char* path, const timeval tv[2]) {
  return futimesat(AT_FDCWD, path, tv, 0);
}

int lutimes(const char* path, const timeval tv[2]) {
  return futimesat(AT_FDCWD, path, tv, AT_SYMLINK_NOFOLLOW);
}

int futimesat(int fd, const char* path, const timeval tv[2]) {
  return futimesat(fd, path, tv, 0);
}

int futimes(int fd, const timeval tv[2]) {
  timespec ts[2];
  if (tv && (!timespec_from_timeval(ts[0], tv[0]) || !timespec_from_timeval(ts[1], tv[1]))) {
    errno = EINVAL;
    return -1;
  }
  return futimens(fd, tv ? ts : nullptr);
}
