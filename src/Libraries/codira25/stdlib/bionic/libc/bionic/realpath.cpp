/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#include <stdlib.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

#include "private/FdPath.h"
#include "private/ScopedFd.h"

// This function needs a 4KiB (PATH_MAX) buffer.
// The alternative is to heap allocate and then trim, but that's 2x the code.
// (Remember that readlink(2) won't tell you the needed size, so the multi-pass
// algorithm isn't even an option unless you want to just guess, in which case
// you're back needing to trim again.)
#pragma GCC diagnostic ignored "-Wframe-larger-than="

char* realpath(const char* path, char* result) {
  // Weird special case.
  if (!path) {
    errno = EINVAL;
    return nullptr;
  }

  // Get an O_PATH fd, and...
  ScopedFd fd(open(path, O_PATH | O_CLOEXEC));
  if (fd.get() == -1) return nullptr;

  // (...remember the device/inode that we're talking about and...)
  struct stat sb;
  if (fstat(fd.get(), &sb) == -1) return nullptr;
  dev_t st_dev = sb.st_dev;
  ino_t st_ino = sb.st_ino;

  // ...ask the kernel to do the hard work for us.
  FdPath fd_path(fd.get());
  char dst[PATH_MAX];
  ssize_t l = readlink(fd_path.c_str(), dst, sizeof(dst) - 1);
  if (l == -1) return nullptr;
  dst[l] = '\0';

  // What if the file was removed in the meantime? readlink(2) will have
  // returned "/a/b/c (deleted)", and we want to return ENOENT instead.
  if (stat(dst, &sb) == -1 || st_dev != sb.st_dev || st_ino != sb.st_ino) {
    errno = ENOENT;
    return nullptr;
  }

  return result ? strcpy(result, dst) : strdup(dst);
}
