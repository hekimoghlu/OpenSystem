/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include <malloc.h>
#include <sys/param.h>
#include <unistd.h>

#include <private/MallocXmlElem.h>

#include "h_malloc.h"

__BEGIN_DECLS
int h_malloc_info(int options, FILE* fp);
__END_DECLS

int h_malloc_info(int options, FILE* fp) {
  if (options != 0) {
    errno = EINVAL;
    return -1;
  }

  fflush(fp);
  int fd = fileno(fp);
  MallocXmlElem root(fd, "malloc", "version=\"jemalloc-1\"");

  // Dump all of the large allocations in the arenas.
  for (size_t i = 0; i < h_mallinfo_narenas(); i++) {
    struct mallinfo mi = h_mallinfo_arena_info(i);
    if (mi.hblkhd != 0) {
      MallocXmlElem arena_elem(fd, "heap", "nr=\"%d\"", i);
      {
        MallocXmlElem(fd, "allocated-large").Contents("%zu", mi.ordblks);
        MallocXmlElem(fd, "allocated-huge").Contents("%zu", mi.uordblks);
        MallocXmlElem(fd, "allocated-bins").Contents("%zu", mi.fsmblks);

        size_t total = 0;
        for (size_t j = 0; j < h_mallinfo_nbins(); j++) {
          struct mallinfo mi = h_mallinfo_bin_info(i, j);
          if (mi.ordblks != 0) {
            MallocXmlElem bin_elem(fd, "bin", "nr=\"%d\"", j);
            MallocXmlElem(fd, "allocated").Contents("%zu", mi.ordblks);
            MallocXmlElem(fd, "nmalloc").Contents("%zu", mi.uordblks);
            MallocXmlElem(fd, "ndalloc").Contents("%zu", mi.fordblks);
            total += mi.ordblks;
          }
        }
        MallocXmlElem(fd, "bins-total").Contents("%zu", total);
      }
    }
  }

  return 0;
}
