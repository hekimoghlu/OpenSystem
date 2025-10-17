/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#include <libgen.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/cdefs.h>
#include <sys/param.h>

#include "bionic/pthread_internal.h"

static int __basename_r(const char* path, char* buffer, size_t buffer_size) {
  const char* startp = nullptr;
  const char* endp = nullptr;
  int len;
  int result;

  // Empty or NULL string gets treated as ".".
  if (path == nullptr || *path == '\0') {
    startp = ".";
    len = 1;
    goto Exit;
  }

  // Strip trailing slashes.
  endp = path + strlen(path) - 1;
  while (endp > path && *endp == '/') {
    endp--;
  }

  // All slashes becomes "/".
  if (endp == path && *endp == '/') {
    startp = "/";
    len = 1;
    goto Exit;
  }

  // Find the start of the base.
  startp = endp;
  while (startp > path && *(startp - 1) != '/') {
    startp--;
  }

  len = endp - startp +1;

 Exit:
  result = len;
  if (buffer == nullptr) {
    return result;
  }
  if (len > static_cast<int>(buffer_size) - 1) {
    len = buffer_size - 1;
    result = -1;
    errno = ERANGE;
  }

  if (len >= 0) {
    memcpy(buffer, startp, len);
    buffer[len] = 0;
  }
  return result;
}

// Since this is a non-standard symbol, it might be hijacked by a basename_r in the executable.
__LIBC32_LEGACY_PUBLIC__ int basename_r(const char* path, char* buffer, size_t buffer_size) {
  return __basename_r(path, buffer, buffer_size);
}

static int __dirname_r(const char* path, char* buffer, size_t buffer_size) {
  const char* endp = nullptr;
  int len;
  int result;

  // Empty or NULL string gets treated as ".".
  if (path == nullptr || *path == '\0') {
    path = ".";
    len = 1;
    goto Exit;
  }

  // Strip trailing slashes.
  endp = path + strlen(path) - 1;
  while (endp > path && *endp == '/') {
    endp--;
  }

  // Find the start of the dir.
  while (endp > path && *endp != '/') {
    endp--;
  }

  // Either the dir is "/" or there are no slashes.
  if (endp == path) {
    path = (*endp == '/') ? "/" : ".";
    len = 1;
    goto Exit;
  }

  do {
    endp--;
  } while (endp > path && *endp == '/');

  len = endp - path + 1;

 Exit:
  result = len;
  if (len + 1 > MAXPATHLEN) {
    errno = ENAMETOOLONG;
    return -1;
  }
  if (buffer == nullptr) {
    return result;
  }

  if (len > static_cast<int>(buffer_size) - 1) {
    len = buffer_size - 1;
    result = -1;
    errno = ERANGE;
  }

  if (len >= 0) {
    memcpy(buffer, path, len);
    buffer[len] = 0;
  }
  return result;
}

// Since this is a non-standard symbol, it might be hijacked by a basename_r in the executable.
__LIBC32_LEGACY_PUBLIC__ int dirname_r(const char* path, char* buffer, size_t buffer_size) {
  return __dirname_r(path, buffer, buffer_size);
}

char* basename(const char* path) {
  char* buf = __get_bionic_tls().basename_buf;
  int rc = __basename_r(path, buf, sizeof(__get_bionic_tls().basename_buf));
  return (rc < 0) ? nullptr : buf;
}

char* dirname(const char* path) {
  char* buf = __get_bionic_tls().dirname_buf;
  int rc = __dirname_r(path, buf, sizeof(__get_bionic_tls().dirname_buf));
  return (rc < 0) ? nullptr : buf;
}
