/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32)

#include "CNIOWindows.h"

#include <assert.h>

int CNIOWindows_sendmmsg(SOCKET s, CNIOWindows_mmsghdr *msgvec, unsigned int vlen,
                         int flags) {
  assert(!"sendmmsg not implemented");
  abort();
}

int CNIOWindows_recvmmsg(SOCKET s, CNIOWindows_mmsghdr *msgvec,
                         unsigned int vlen, int flags,
                         struct timespec *timeout) {
  assert(!"recvmmsg not implemented");
  abort();
}

const void *CNIOWindows_CMSG_DATA(const WSACMSGHDR *pcmsg) {
  return WSA_CMSG_DATA(pcmsg);
}

void *CNIOWindows_CMSG_DATA_MUTABLE(LPWSACMSGHDR pcmsg) {
  return WSA_CMSG_DATA(pcmsg);
}

WSACMSGHDR *CNIOWindows_CMSG_FIRSTHDR(const WSAMSG *msg) {
  return WSA_CMSG_FIRSTHDR(msg);
}

WSACMSGHDR *CNIOWindows_CMSG_NXTHDR(const WSAMSG *msg, LPWSACMSGHDR cmsg) {
  return WSA_CMSG_NXTHDR(msg, cmsg);
}

size_t CNIOWindows_CMSG_LEN(size_t length) {
  return WSA_CMSG_LEN(length);
}

size_t CNIOWindows_CMSG_SPACE(size_t length) {
  return WSA_CMSG_SPACE(length);
}

#endif
