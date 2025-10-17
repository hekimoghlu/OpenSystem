/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#define MAP_FILE 0
#endif

#include "filemap.h"

int filemap(const char *name,
	    void (*processor)(const void *, size_t, const char *, void *arg),
	    void *arg)
{
  int fd;
  size_t nbytes;
  struct stat sb;
  void *p;

  fd = open(name, O_RDONLY);
  if (fd < 0) {
    perror(name);
    return 0;
  }
  if (fstat(fd, &sb) < 0) {
    perror(name);
    close(fd);
    return 0;
  }
  if (!S_ISREG(sb.st_mode)) {
    close(fd);
    fprintf(stderr, "%s: not a regular file\n", name);
    return 0;
  }
  
  nbytes = sb.st_size;
  p = (void *)mmap((caddr_t)0, (size_t)nbytes, PROT_READ,
		   MAP_FILE|MAP_PRIVATE, fd, (off_t)0);
  if (p == (void *)-1) {
    perror(name);
    close(fd);
    return 0;
  }
  processor(p, nbytes, name, arg);
  munmap((caddr_t)p, nbytes);
  close(fd);
  return 1;
}
