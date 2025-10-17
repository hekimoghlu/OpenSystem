/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#include <sys/signalfd.h>

#include "private/SigSetConverter.h"

extern "C" int __signalfd4(int, const sigset64_t*, size_t, int);

int signalfd64(int fd, const sigset64_t* mask, int flags) {
  return __signalfd4(fd, mask, sizeof(*mask), flags);
}

int signalfd(int fd, const sigset_t* mask, int flags) {
  // The underlying `__signalfd4` system call only takes `sigset64_t`.
  SigSetConverter set{mask};
  return signalfd64(fd, set.ptr, flags);
}
