/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include <unistd.h>
#include <errno.h>

extern "C" int __faccessat(int, const char*, int);

int faccessat(int dirfd, const char* pathname, int mode, int flags) {
  // "The mode specifies the accessibility check(s) to be performed,
  // and is either the value F_OK, or a mask consisting of the
  // bitwise OR of one or more of R_OK, W_OK, and X_OK."
  if ((mode != F_OK) && ((mode & ~(R_OK | W_OK | X_OK)) != 0) &&
      ((mode & (R_OK | W_OK | X_OK)) == 0)) {
    errno = EINVAL;
    return -1;
  }

  if (flags != 0) {
    // We deliberately don't support AT_SYMLINK_NOFOLLOW, a glibc
    // only feature which is error prone and dangerous.
    // More details at http://permalink.gmane.org/gmane.linux.lib.musl.general/6952
    //
    // AT_EACCESS isn't supported either. Android doesn't have setuid
    // programs, and never runs code with euid!=uid.
    //
    // We could use faccessat2(2) from Linux 5.8, but since we don't want the
    // first feature and don't need the second, we just reject such requests.
    errno = EINVAL;
    return -1;
  }

  return __faccessat(dirfd, pathname, mode);
}
