/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#ifndef __ASM_GENERIC_POLL_H
#define __ASM_GENERIC_POLL_H
#define POLLIN 0x0001
#define POLLPRI 0x0002
#define POLLOUT 0x0004
#define POLLERR 0x0008
#define POLLHUP 0x0010
#define POLLNVAL 0x0020
#define POLLRDNORM 0x0040
#define POLLRDBAND 0x0080
#ifndef POLLWRNORM
#define POLLWRNORM 0x0100
#endif
#ifndef POLLWRBAND
#define POLLWRBAND 0x0200
#endif
#ifndef POLLMSG
#define POLLMSG 0x0400
#endif
#ifndef POLLREMOVE
#define POLLREMOVE 0x1000
#endif
#ifndef POLLRDHUP
#define POLLRDHUP 0x2000
#endif
#define POLLFREE ( __poll_t) 0x4000
#define POLL_BUSY_LOOP ( __poll_t) 0x8000
struct pollfd {
  int fd;
  short events;
  short revents;
};
#endif
