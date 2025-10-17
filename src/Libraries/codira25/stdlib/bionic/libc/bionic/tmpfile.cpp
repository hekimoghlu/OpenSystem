/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "private/ErrnoRestorer.h"

static FILE* __fd_to_fp(int fd) {
  FILE* fp = fdopen(fd, "w+");
  if (fp != nullptr) return fp;

  ErrnoRestorer errno_restorer;
  close(fd);
  return nullptr;
}

// O_TMPFILE isn't available until Linux 3.11, so we fall back to this on
// older kernels. AOSP was on a new enough kernel in the Lollipop timeframe,
// so this code should be obsolete by 2025.
static FILE* __tmpfile_dir_legacy(const char* tmp_dir) {
  char* path = nullptr;
  if (asprintf(&path, "%s/tmp.XXXXXXXXXX", tmp_dir) == -1) {
    return nullptr;
  }

  int fd = mkstemp(path);
  if (fd == -1) {
    free(path);
    return nullptr;
  }

  // Unlink the file now so that it's removed when closed.
  unlink(path);
  free(path);

  // Can we still use the file now it's unlinked?
  // File systems without hard link support won't have the usual Unix semantics.
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    ErrnoRestorer errno_restorer;
    close(fd);
    return nullptr;
  }

  return __fd_to_fp(fd);
}

const char* __get_TMPDIR() {
  // Use $TMPDIR if set, or fall back to /data/local/tmp otherwise.
  // Useless for apps, but good enough for the shell.
  const char* tmpdir = getenv("TMPDIR");
  return (tmpdir == nullptr) ? "/data/local/tmp" : tmpdir;
}

FILE* tmpfile() {
  const char* tmpdir = __get_TMPDIR();
  int fd = open(tmpdir, O_TMPFILE | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) return __tmpfile_dir_legacy(tmpdir);
  return __fd_to_fp(fd);
}
__strong_alias(tmpfile64, tmpfile);

char* tempnam(const char* dir, const char* prefix) {
  // This function is a terrible idea, marked deprecated in our headers,
  // and marked obsolescent by POSIX.1-2008, but we make some effort anyway
  // since we can't easily remove it...

  // $TMPDIR overrides any directory passed in.
  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir != nullptr) dir = tmpdir;

  // If we still have no directory, we'll give you a default.
  // It's useless for apps, but good enough for the shell.
  if (dir == nullptr) dir = "/data/local/tmp";

  // Default prefix?
  if (prefix == nullptr) prefix = "tempnam.";

  // Make up a mktemp(3) template and defer to it for the real work.
  char* path = nullptr;
  if (asprintf(&path, "%s/%sXXXXXXXXXX", dir, prefix) == -1) return nullptr;
  if (mktemp(path) == nullptr) {
    free(path);
    return nullptr;
  }
  return path;
}

char* tmpnam(char* s) {
  // This function is a terrible idea, marked deprecated in our headers,
  // and marked obsolescent by POSIX-1.2008, but we make some effort anyway
  // since we can't easily remove it...

  // Default buffer?
  static char buf[L_tmpnam];
  if (s == nullptr) s = buf;

  // Make up a mktemp(3) template and defer to it for the real work.
  snprintf(s, L_tmpnam, "%s/tmpnam.XXXXXXXXXX", __get_TMPDIR());
  return mktemp(s);
}
