/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
#include <fcntl.h>

#if __arm__
// Only arm32 is missing the sync_file_range() syscall,
// and needs us to manually re-order arguments for it.
// (Because arm32 needs register pairs for 64-bit values to start on an even register.)
extern "C" int __sync_file_range2(int, unsigned int, off64_t, off64_t);
int sync_file_range(int fd, off64_t offset, off64_t length, unsigned int flags) {
  return __sync_file_range2(fd, flags, offset, length);
}
#endif
