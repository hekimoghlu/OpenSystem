/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include <sys/eventfd.h>
#include <unistd.h>

#include "private/bionic_fdtrack.h"

extern "C" int __eventfd(unsigned int initval, int flags);

int eventfd(unsigned int initval, int flags) {
  return FDTRACK_CREATE(__eventfd(initval, flags));
}

int eventfd_read(int fd, eventfd_t* value) {
  return (read(fd, value, sizeof(*value)) == sizeof(*value)) ? 0 : -1;
}

int eventfd_write(int fd, eventfd_t value) {
  return (write(fd, &value, sizeof(value)) == sizeof(value)) ? 0 : -1;
}
