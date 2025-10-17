/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <ffi.h>
#include <fficonfig.h>

#define MAX_ARGS 256

#define CHECK(x) !(x) ? abort() : 0


/* Prefer MAP_ANON(YMOUS) to /dev/zero, since we don't need to keep a
   file open.  */
#ifdef HAVE_MMAP_ANON
# undef HAVE_MMAP_DEV_ZERO

# include <sys/mman.h>
# ifndef MAP_FAILED
#  define MAP_FAILED -1
# endif
# if !defined (MAP_ANONYMOUS) && defined (MAP_ANON)
#  define MAP_ANONYMOUS MAP_ANON
# endif
# define USING_MMAP

#endif

#ifdef HAVE_MMAP_DEV_ZERO

# include <sys/mman.h>
# ifndef MAP_FAILED
#  define MAP_FAILED -1
# endif
# define USING_MMAP

#endif

#ifdef USING_MMAP
static inline void *
allocate_mmap (size_t size)
{
  void *page;
#if defined (HAVE_MMAP_DEV_ZERO)
  static int dev_zero_fd = -1;
#endif

#ifdef HAVE_MMAP_DEV_ZERO
  if (dev_zero_fd == -1)
    {
      dev_zero_fd = open ("/dev/zero", O_RDONLY);
      if (dev_zero_fd == -1)
	{
	  perror ("open /dev/zero: %m");
	  exit (1);
	}
    }
#endif


#ifdef HAVE_MMAP_ANON
  page = mmap (NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
	       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
#ifdef HAVE_MMAP_DEV_ZERO
  page = mmap (NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
	       MAP_PRIVATE, dev_zero_fd, 0);
#endif

  if (page == (void *) MAP_FAILED)
    {
      perror ("virtual memory exhausted");
      exit (1);
    }

  return page;
}

#endif
