/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#include <sys/epoll.h>

#include "private/SigSetConverter.h"
#include "private/bionic_fdtrack.h"

extern "C" int __epoll_create1(int flags);
extern "C" int __epoll_pwait(int, epoll_event*, int, int, const sigset64_t*, size_t);
extern "C" int __epoll_pwait2(int, epoll_event*, int, const __kernel_timespec*, const sigset64_t*,
                              size_t);

int epoll_create(int size) {
  if (size <= 0) {
    errno = EINVAL;
    return -1;
  }
  return FDTRACK_CREATE(__epoll_create1(0));
}

int epoll_create1(int flags) {
  return FDTRACK_CREATE(__epoll_create1(flags));
}

int epoll_pwait(int fd, epoll_event* events, int max_events, int timeout, const sigset_t* ss) {
  SigSetConverter set{ss};
  return epoll_pwait64(fd, events, max_events, timeout, set.ptr);
}

int epoll_pwait64(int fd, epoll_event* events, int max_events, int timeout, const sigset64_t* ss) {
  return __epoll_pwait(fd, events, max_events, timeout, ss, sizeof(*ss));
}

int epoll_pwait2(int fd, epoll_event* events, int max_events, const timespec* timeout,
                 const sigset_t* ss) {
  SigSetConverter set{ss};
  return epoll_pwait2_64(fd, events, max_events, timeout, set.ptr);
}

int epoll_pwait2_64(int fd, epoll_event* events, int max_events, const timespec* timeout,
                    const sigset64_t* ss) {
  // epoll_pwait2() is our first syscall that assumes a 64-bit time_t even for
  // 32-bit processes, so for ILP32 we need to convert.
  // TODO: factor this out into a TimeSpecConverter as/when we get more syscalls like this.
#if __LP64__
  const __kernel_timespec* kts_ptr = reinterpret_cast<const __kernel_timespec*>(timeout);
#else
  __kernel_timespec kts;
  const __kernel_timespec* kts_ptr = nullptr;
  if (timeout) {
    kts.tv_sec = timeout->tv_sec;
    kts.tv_nsec = timeout->tv_nsec;
    kts_ptr = &kts;
  }
#endif
  return __epoll_pwait2(fd, events, max_events, kts_ptr, ss, sizeof(*ss));
}

int epoll_wait(int fd, struct epoll_event* events, int max_events, int timeout) {
  return epoll_pwait64(fd, events, max_events, timeout, nullptr);
}
