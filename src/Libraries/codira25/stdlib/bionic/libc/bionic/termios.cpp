/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#include <termios.h>
#include <unistd.h>

// Most of termios was missing in the platform until L, but available as inlines in the NDK.
// We share definitions with the NDK to avoid bugs (https://github.com/android-ndk/ndk/issues/441).
#define __BIONIC_TERMIOS_INLINE /* Out of line. */
#include <bits/termios_inlines.h>

// POSIX added a couple more functions much later, so do the same for them.
#define __BIONIC_TERMIOS_WINSIZE_INLINE /* Out of line. */
#include <bits/termios_winsize_inlines.h>

// Actually declared in <unistd.h>, present on all API levels.
pid_t tcgetpgrp(int fd) {
  pid_t pid;
  return (ioctl(fd, TIOCGPGRP, &pid) == -1) ? -1 : pid;
}

// Actually declared in <unistd.h>, present on all API levels.
int tcsetpgrp(int fd, pid_t pid) {
  return ioctl(fd, TIOCSPGRP, &pid);
}
