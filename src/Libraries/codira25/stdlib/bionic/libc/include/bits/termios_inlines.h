/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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
#ifndef _BITS_TERMIOS_INLINES_H_
#define _BITS_TERMIOS_INLINES_H_

#include <sys/cdefs.h>

#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include <linux/termios.h>

#if !defined(__BIONIC_TERMIOS_INLINE)
#define __BIONIC_TERMIOS_INLINE static __inline
#endif

__BEGIN_DECLS

// Supporting separate input and output speeds would require an ABI
// change for `struct termios`.

static __inline speed_t cfgetspeed(const struct termios* _Nonnull s) {
  return __BIONIC_CAST(static_cast, speed_t, s->c_cflag & CBAUD);
}

__BIONIC_TERMIOS_INLINE speed_t cfgetispeed(const struct termios* _Nonnull s) {
  return cfgetspeed(s);
}

__BIONIC_TERMIOS_INLINE speed_t cfgetospeed(const struct termios* _Nonnull s) {
  return cfgetspeed(s);
}

__BIONIC_TERMIOS_INLINE void cfmakeraw(struct termios* _Nonnull s) {
  s->c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL|IXON);
  s->c_oflag &= ~OPOST;
  s->c_lflag &= ~(ECHO|ECHONL|ICANON|ISIG|IEXTEN);
  s->c_cflag &= ~(CSIZE|PARENB);
  s->c_cflag |= CS8;
  s->c_cc[VMIN] = 1;
  s->c_cc[VTIME] = 0;
}

__BIONIC_TERMIOS_INLINE int cfsetspeed(struct termios* _Nonnull s, speed_t speed) {
  // CBAUD is 0x100f, and every matching bit pattern has a Bxxx constant.
  if ((speed & ~CBAUD) != 0) {
    errno = EINVAL;
    return -1;
  }
  s->c_cflag = (s->c_cflag & ~CBAUD) | (speed & CBAUD);
  return 0;
}

__BIONIC_TERMIOS_INLINE int cfsetispeed(struct termios* _Nonnull s, speed_t speed) {
  return cfsetspeed(s, speed);
}

__BIONIC_TERMIOS_INLINE int cfsetospeed(struct termios* _Nonnull s, speed_t speed) {
  return cfsetspeed(s, speed);
}

__BIONIC_TERMIOS_INLINE int tcdrain(int fd) {
  // A non-zero argument to TCSBRK means "don't send a break".
  // The drain is a side-effect of the ioctl!
  return ioctl(fd, TCSBRK, __BIONIC_CAST(static_cast, unsigned long, 1));
}

__BIONIC_TERMIOS_INLINE int tcflow(int fd, int action) {
  return ioctl(fd, TCXONC, __BIONIC_CAST(static_cast, unsigned long, action));
}

__BIONIC_TERMIOS_INLINE int tcflush(int fd, int queue) {
  return ioctl(fd, TCFLSH, __BIONIC_CAST(static_cast, unsigned long, queue));
}

__BIONIC_TERMIOS_INLINE int tcgetattr(int fd, struct termios* _Nonnull s) {
  return ioctl(fd, TCGETS, s);
}

__BIONIC_TERMIOS_INLINE pid_t tcgetsid(int fd) {
  pid_t sid;
  return (ioctl(fd, TIOCGSID, &sid) == -1) ? -1 : sid;
}

__BIONIC_TERMIOS_INLINE int tcsendbreak(int fd, int duration) {
  return ioctl(fd, TCSBRKP, __BIONIC_CAST(static_cast, unsigned long, duration));
}

__BIONIC_TERMIOS_INLINE int tcsetattr(int fd, int optional_actions, const struct termios* _Nonnull s) {
  int cmd;
  switch (optional_actions) {
    case TCSANOW: cmd = TCSETS; break;
    case TCSADRAIN: cmd = TCSETSW; break;
    case TCSAFLUSH: cmd = TCSETSF; break;
    default: errno = EINVAL; return -1;
  }
  return ioctl(fd, cmd, s);
}

__END_DECLS

#endif
