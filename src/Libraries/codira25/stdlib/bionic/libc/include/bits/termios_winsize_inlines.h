/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#pragma once

#include <sys/cdefs.h>

#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include <linux/termios.h>

#if !defined(__BIONIC_TERMIOS_WINSIZE_INLINE)
#define __BIONIC_TERMIOS_WINSIZE_INLINE static __inline
#endif

__BEGIN_DECLS

__BIONIC_TERMIOS_WINSIZE_INLINE int tcgetwinsize(int __fd, struct winsize* _Nonnull __size) {
  return ioctl(__fd, TIOCGWINSZ, __size);
}

__BIONIC_TERMIOS_WINSIZE_INLINE int tcsetwinsize(int __fd, const struct winsize* _Nonnull __size) {
  return ioctl(__fd, TIOCSWINSZ, __size);
}

__END_DECLS
