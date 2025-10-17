/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include <unistd.h>

#include "private/bionic_fdtrack.h"

extern "C" int __pipe2(int pipefd[2], int flags);

int pipe(int pipefd[2]) {
  int rc = __pipe2(pipefd, 0);
  if (rc == 0) {
    FDTRACK_CREATE(pipefd[0]);
    FDTRACK_CREATE(pipefd[1]);
  }
  return rc;
}

int pipe2(int pipefd[2], int flags) {
  int rc = __pipe2(pipefd, flags);
  if (rc == 0) {
    FDTRACK_CREATE(pipefd[0]);
    FDTRACK_CREATE(pipefd[1]);
  }
  return rc;
}
