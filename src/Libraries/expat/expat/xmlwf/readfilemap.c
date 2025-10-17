/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>

/* Functions close(2) and read(2) */
#if ! defined(_WIN32) && ! defined(_WIN64)
#  include <unistd.h>
#endif

/* Function "read": */
#if defined(_MSC_VER)
#  include <io.h>
/* https://msdn.microsoft.com/en-us/library/wyssk1bs(v=vs.100).aspx */
#  define EXPAT_read _read
#  define EXPAT_read_count_t int
#  define EXPAT_read_req_t unsigned int
#else /* POSIX */
/* https://pubs.opengroup.org/onlinepubs/009695399/functions/read.html */
#  define EXPAT_read read
#  define EXPAT_read_count_t ssize_t
#  define EXPAT_read_req_t size_t
#endif

#ifndef S_ISREG
#  ifndef S_IFREG
#    define S_IFREG _S_IFREG
#  endif
#  ifndef S_IFMT
#    define S_IFMT _S_IFMT
#  endif
#  define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif /* not S_ISREG */

#ifndef O_BINARY
#  ifdef _O_BINARY
#    define O_BINARY _O_BINARY
#  else
#    define O_BINARY 0
#  endif
#endif

#include "xmltchar.h"
#include "filemap.h"

int
filemap(const tchar *name,
        void (*processor)(const void *, size_t, const tchar *, void *arg),
        void *arg) {
  size_t nbytes;
  int fd;
  EXPAT_read_count_t n;
  struct stat sb;
  void *p;

  fd = topen(name, O_RDONLY | O_BINARY);
  if (fd < 0) {
    tperror(name);
    return 0;
  }
  if (fstat(fd, &sb) < 0) {
    tperror(name);
    close(fd);
    return 0;
  }
  if (! S_ISREG(sb.st_mode)) {
    ftprintf(stderr, T("%s: not a regular file\n"), name);
    close(fd);
    return 0;
  }
  if (sb.st_size > XML_MAX_CHUNK_LEN) {
    close(fd);
    return 2; /* Cannot be passed to XML_Parse in one go */
  }

  nbytes = sb.st_size;
  /* malloc will return NULL with nbytes == 0, handle files with size 0 */
  if (nbytes == 0) {
    static const char c = '\0';
    processor(&c, 0, name, arg);
    close(fd);
    return 1;
  }
  p = malloc(nbytes);
  if (! p) {
    ftprintf(stderr, T("%s: out of memory\n"), name);
    close(fd);
    return 0;
  }
  n = EXPAT_read(fd, p, (EXPAT_read_req_t)nbytes);
  if (n < 0) {
    tperror(name);
    free(p);
    close(fd);
    return 0;
  }
  if (n != (EXPAT_read_count_t)nbytes) {
    ftprintf(stderr, T("%s: read unexpected number of bytes\n"), name);
    free(p);
    close(fd);
    return 0;
  }
  processor(p, nbytes, name, arg);
  free(p);
  close(fd);
  return 1;
}

