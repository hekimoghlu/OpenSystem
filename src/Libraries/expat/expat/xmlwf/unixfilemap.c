/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#ifndef MAP_FILE
#  define MAP_FILE 0
#endif

#include "xmltchar.h"
#include "filemap.h"

#ifdef XML_UNICODE_WCHAR_T
#  define XML_FMT_STR "ls"
#else
#  define XML_FMT_STR "s"
#endif

int
filemap(const tchar *name,
        void (*processor)(const void *, size_t, const tchar *, void *arg),
        void *arg) {
  int fd;
  size_t nbytes;
  struct stat sb;
  void *p;

  fd = topen(name, O_RDONLY);
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
    close(fd);
    fprintf(stderr, "%" XML_FMT_STR ": not a regular file\n", name);
    return 0;
  }
  if (sb.st_size > XML_MAX_CHUNK_LEN) {
    close(fd);
    return 2; /* Cannot be passed to XML_Parse in one go */
  }

  nbytes = sb.st_size;
  /* mmap fails for zero length files */
  if (nbytes == 0) {
    static const char c = '\0';
    processor(&c, 0, name, arg);
    close(fd);
    return 1;
  }
  p = (void *)mmap((void *)0, (size_t)nbytes, PROT_READ, MAP_FILE | MAP_PRIVATE,
                   fd, (off_t)0);
  if (p == (void *)-1) {
    tperror(name);
    close(fd);
    return 0;
  }
  processor(p, nbytes, name, arg);
  munmap((void *)p, nbytes);
  close(fd);
  return 1;
}

