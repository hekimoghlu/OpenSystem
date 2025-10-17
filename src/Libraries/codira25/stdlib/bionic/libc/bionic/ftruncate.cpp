/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include <sys/cdefs.h>
#include <unistd.h>

#if !defined(__LP64__)
static_assert(sizeof(off_t) == 4,
              "libc can't be built with _FILE_OFFSET_BITS=64.");

// The kernel's implementation of ftruncate uses an unsigned long for the length
// parameter, so it will not catch negative values. On the other hand
// ftruncate64 does check for this, so just forward the call.
int ftruncate(int filedes, off_t length) {
  return ftruncate64(filedes, length);
}
#endif  // !defined(__LP64__)
