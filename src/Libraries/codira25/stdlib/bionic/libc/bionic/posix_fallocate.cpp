/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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

#include "private/ErrnoRestorer.h"

int posix_fallocate(int fd, off_t offset, off_t length) {
  ErrnoRestorer errno_restorer;
  return (fallocate(fd, 0, offset, length) == 0) ? 0 : errno;
}

int posix_fallocate64(int fd, off64_t offset, off64_t length) {
  ErrnoRestorer errno_restorer;
  return (fallocate64(fd, 0, offset, length) == 0) ? 0 : errno;
}
