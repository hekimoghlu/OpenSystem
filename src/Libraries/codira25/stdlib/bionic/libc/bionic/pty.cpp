/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
#include <fcntl.h>
#include <pty.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <utmp.h>

#include "bionic/pthread_internal.h"
#include "private/FdPath.h"

int getpt() {
  return posix_openpt(O_RDWR|O_NOCTTY);
}

int grantpt(int) {
  return 0;
}

int posix_openpt(int flags) {
  return open("/dev/ptmx", flags);
}

char* ptsname(int fd) {
  bionic_tls& tls = __get_bionic_tls();
  char* buf = tls.ptsname_buf;
  int error = ptsname_r(fd, buf, sizeof(tls.ptsname_buf));
  return (error == 0) ? buf : nullptr;
}

int ptsname_r(int fd, char* buf, size_t len) {
  if (buf == nullptr) {
    errno = EINVAL;
    return errno;
  }

  unsigned int pty_num;
  if (ioctl(fd, TIOCGPTN, &pty_num) != 0) {
    errno = ENOTTY;
    return errno;
  }

  if (snprintf(buf, len, "/dev/pts/%u", pty_num) >= static_cast<int>(len)) {
    errno = ERANGE;
    return errno;
  }

  return 0;
}

char* ttyname(int fd) {
  bionic_tls& tls = __get_bionic_tls();
  char* buf = tls.ttyname_buf;
  int error = ttyname_r(fd, buf, sizeof(tls.ttyname_buf));
  return (error == 0) ? buf : nullptr;
}

int ttyname_r(int fd, char* buf, size_t len) {
  if (buf == nullptr) {
    errno = EINVAL;
    return errno;
  }

  if (!isatty(fd)) {
    return errno;
  }

  ssize_t count = readlink(FdPath(fd).c_str(), buf, len);
  if (count == -1) {
    return errno;
  }
  if (static_cast<size_t>(count) == len) {
    errno = ERANGE;
    return errno;
  }
  buf[count] = '\0';
  return 0;
}

int unlockpt(int fd) {
  int unlock = 0;
  return ioctl(fd, TIOCSPTLCK, &unlock);
}

int openpty(int* pty, int* tty, char* name, const termios* t, const winsize* ws) {
  *pty = getpt();
  if (*pty == -1) {
    return -1;
  }

  if (grantpt(*pty) == -1 || unlockpt(*pty) == -1) {
    close(*pty);
    return -1;
  }

  char buf[32];
  if (name == nullptr) {
    name = buf;
  }
  if (ptsname_r(*pty, name, sizeof(buf)) != 0) {
    close(*pty);
    return -1;
  }

  *tty = open(name, O_RDWR | O_NOCTTY);
  if (*tty == -1) {
    close(*pty);
    return -1;
  }

  if (t != nullptr) {
    tcsetattr(*tty, TCSAFLUSH, t);
  }
  if (ws != nullptr) {
    ioctl(*tty, TIOCSWINSZ, ws);
  }

  return 0;
}

int forkpty(int* parent_pty, char* child_tty_name, const termios* t, const winsize* ws) {
  int pty;
  int tty;
  if (openpty(&pty, &tty, child_tty_name, t, ws) == -1) {
    return -1;
  }

  pid_t pid = fork();
  if (pid == -1) {
    close(pty);
    close(tty);
    return -1;
  }

  if (pid == 0) {
    // Child.
    *parent_pty = -1;
    close(pty);
    if (login_tty(tty) == -1) {
      _exit(1);
    }
    return 0;
  }

  // Parent.
  *parent_pty = pty;
  close(tty);
  return pid;
}

int login_tty(int fd) {
  setsid();

  if (ioctl(fd, TIOCSCTTY, nullptr) == -1) {
    return -1;
  }

  dup2(fd, STDIN_FILENO);
  dup2(fd, STDOUT_FILENO);
  dup2(fd, STDERR_FILENO);
  if (fd > STDERR_FILENO) {
    close(fd);
  }

  return 0;
}
