/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include <sys/cdefs.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "local.h"

// See https://pubs.opengroup.org/onlinepubs/9799919799.2024edition/functions/fmemopen.html
// and https://man7.org/linux/man-pages/man3/fmemopen.3.html for documentation.

struct fmemopen_cookie {
  char* buf;
  char* allocation;
  size_t capacity;
  size_t size;
  size_t offset;
  bool append;
};

static int fmemopen_read(void* cookie, char* buf, int n) {
  fmemopen_cookie* ck = static_cast<fmemopen_cookie*>(cookie);

  if (static_cast<size_t>(n) > ck->size - ck->offset) n = ck->size - ck->offset;

  if (n > 0) {
    memmove(buf, ck->buf + ck->offset, n);
    ck->offset += n;
  }
  return n;
}

static int fmemopen_write(void* cookie, const char* buf, int n) {
  fmemopen_cookie* ck = static_cast<fmemopen_cookie*>(cookie);

  // We don't need to add the trailing NUL if there's already a trailing NUL
  // in the data we're writing.
  size_t space_for_null = (n > 0 && buf[n - 1] != '\0') ? 1 : 0;

  // Undo any seeking/reading on an append-only stream.
  if (ck->append) ck->offset = ck->size;

  // How much can we actually fit?
  if (static_cast<size_t>(n) + space_for_null > ck->capacity - ck->offset) {
    n = ck->capacity - ck->offset - space_for_null;
    // Give up if we don't even have room for one byte of userdata.
    if (n <= 0) {
      errno = ENOSPC;
      return -1;
    }
  }

  if (n > 0) {
    memmove(ck->buf + ck->offset, buf, n);
    ck->offset += n;
    // Is this the furthest we've ever been?
    if (ck->offset >= ck->size) {
      if (buf[n - 1] != '\0') ck->buf[ck->offset] = '\0';
      ck->size = ck->offset;
    }
  }
  return n;
}

static fpos_t fmemopen_seek(void* cookie, fpos_t offset, int whence) {
  fmemopen_cookie* ck = static_cast<fmemopen_cookie*>(cookie);

  if (whence == SEEK_SET && (offset >= 0 && static_cast<size_t>(offset) <= ck->capacity)) {
    return (ck->offset = offset);
  } else if (whence == SEEK_CUR && (ck->offset + offset <= ck->capacity)) {
    return (ck->offset += offset);
  } else if (whence == SEEK_END && (offset <= 0 && static_cast<size_t>(-offset) <= ck->size)) {
    return (ck->offset = ck->size + offset);
  }
  errno = EINVAL;
  return -1;
}

static int fmemopen_close(void* cookie) {
  fmemopen_cookie* ck = static_cast<fmemopen_cookie*>(cookie);
  free(ck->allocation);
  free(ck);
  return 0;
}

FILE* fmemopen(void* buf, size_t capacity, const char* mode) {
  int flags;
  if (__sflags(mode, &flags) == 0) {
    errno = EINVAL;
    return nullptr;
  }

  fmemopen_cookie* ck = static_cast<fmemopen_cookie*>(calloc(sizeof(fmemopen_cookie), 1));
  if (ck == nullptr) return nullptr;

  ck->buf = static_cast<char*>(buf);
  ck->capacity = capacity;

  if (ck->buf == nullptr) ck->buf = ck->allocation = static_cast<char*>(calloc(capacity, 1));
  if (ck->buf == nullptr) {
    free(ck);
    return nullptr;
  }

  FILE* fp = funopen(ck,
                     (flags & O_WRONLY) ? nullptr : fmemopen_read,
                     (flags & O_RDONLY) ? nullptr : fmemopen_write,
                     fmemopen_seek,
                     fmemopen_close);
  if (fp == nullptr) {
    fmemopen_close(ck);
    return nullptr;
  }

  if (mode[0] == 'a') {
    ck->size = strnlen(ck->buf, ck->capacity);
    ck->offset = ck->size;
    ck->append = true;
  } else if (mode[0] == 'r') {
    ck->size = capacity;
    ck->offset = 0;
  } else if (mode[0] == 'w') {
    ck->size = 0;
    ck->offset = 0;
    if (capacity > 0) {
      ck->buf[0] = '\0';
    }
  }

  return fp;
}
