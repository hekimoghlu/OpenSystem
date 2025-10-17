/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#include <string.h>
#include <unistd.h>

extern "C" int __getcwd(char* buf, size_t size);

char* getcwd(char* buf, size_t size) {
  // You can't specify size 0 unless you're asking us to allocate for you.
  if (buf != nullptr && size == 0) {
    errno = EINVAL;
    return nullptr;
  }

  // Allocate a buffer if necessary.
  char* allocated_buf = nullptr;
  size_t allocated_size = size;
  if (buf == nullptr) {
    if (size == 0) {
      // The Linux kernel won't return more than a page, so translate size 0 to 4KiB.
      // TODO: if we need to support paths longer than that, we'll have to walk the tree ourselves.
      allocated_size = getpagesize();
    }
    buf = allocated_buf = static_cast<char*>(malloc(allocated_size));
    if (buf == nullptr) {
      return nullptr;
    }
  }

  // Ask the kernel to fill our buffer.
  int rc = __getcwd(buf, allocated_size);
  if (rc == -1) {
    free(allocated_buf);
    // __getcwd set errno.
    return nullptr;
  }

  // If we allocated a whole page, only return as large an allocation as necessary.
  if (allocated_buf != nullptr) {
    if (size == 0) {
      buf = strdup(allocated_buf);
      free(allocated_buf);
    } else {
      buf = allocated_buf;
    }
  }

  return buf;
}
