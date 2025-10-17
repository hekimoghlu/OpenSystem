/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
#include <gtest/gtest.h>

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <unistd.h>

#include <thread>

#include "SignalUtils.h"

static void TestSignalFd(int fd, int signal) {
  ASSERT_NE(-1, fd) << strerror(errno);

  ASSERT_EQ(0, raise(signal));

  signalfd_siginfo sfd_si;
  ASSERT_EQ(static_cast<ssize_t>(sizeof(sfd_si)), read(fd, &sfd_si, sizeof(sfd_si)));

  ASSERT_EQ(signal, static_cast<int>(sfd_si.ssi_signo));

  close(fd);
}

TEST(sys_signalfd, signalfd) {
  SignalMaskRestorer smr;

  sigset_t mask = {};
  sigaddset(&mask, SIGALRM);
  ASSERT_EQ(0, sigprocmask(SIG_SETMASK, &mask, nullptr));

  TestSignalFd(signalfd(-1, &mask, SFD_CLOEXEC), SIGALRM);
}

TEST(sys_signalfd, signalfd64) {
#if defined(__BIONIC__)
  SignalMaskRestorer smr;

  sigset64_t mask = {};
  sigaddset64(&mask, SIGRTMIN);
  ASSERT_EQ(0, sigprocmask64(SIG_SETMASK, &mask, nullptr));

  TestSignalFd(signalfd64(-1, &mask, SFD_CLOEXEC), SIGRTMIN);
#endif
}
