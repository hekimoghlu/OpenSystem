/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include <string.h>
#include <sys/socket.h>

#include <async_safe/log.h>

#include "private/bionic_fdtrack.h"

extern "C" ssize_t __recvmsg(int __fd, struct msghdr* __msg, int __flags);
extern "C" int __recvmmsg(int __fd, struct mmsghdr* __msgs, unsigned int __msg_count, int __flags,
                          const struct timespec* __timeout);

static inline __attribute__((artificial)) __attribute__((always_inline)) void track_fds(
    struct msghdr* msg, const char* function_name) {
  if (!atomic_load(&__android_fdtrack_hook)) {
    return;
  }

  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(msg); cmsg; cmsg = CMSG_NXTHDR(msg, cmsg)) {
    if (cmsg->cmsg_type != SCM_RIGHTS) {
      continue;
    }

    if (cmsg->cmsg_len <= sizeof(struct cmsghdr)) {
      continue;
    }

    size_t data_length = cmsg->cmsg_len - sizeof(struct cmsghdr);
    if (data_length % sizeof(int) != 0) {
      async_safe_fatal("invalid cmsg length: %zu", data_length);
    }

    for (size_t offset = 0; offset < data_length; offset += sizeof(int)) {
      int fd;
      memcpy(&fd, CMSG_DATA(cmsg) + offset, sizeof(int));
      FDTRACK_CREATE_NAME(function_name, fd);
    }
  }
}

ssize_t recvmsg(int __fd, struct msghdr* __msg, int __flags) {
  ssize_t rc = __recvmsg(__fd, __msg, __flags);
  if (rc == -1) {
    return -1;
  }
  track_fds(__msg, "recvmsg");
  return rc;
}

int recvmmsg(int __fd, struct mmsghdr* __msgs, unsigned int __msg_count, int __flags,
             const struct timespec* __timeout) {
  int rc = __recvmmsg(__fd, __msgs, __msg_count, __flags, __timeout);
  if (rc == -1) {
    return -1;
  }
  for (int i = 0; i < rc; ++i) {
    track_fds(&__msgs[i].msg_hdr, "recvmmsg");
  }
  return rc;
}
