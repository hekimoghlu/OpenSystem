/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#include "common.h"
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif

RCSID("$Id$");

void
set_buffer_size(int fd, int read)
{
#if defined(SO_RCVBUF) && defined(SO_SNDBUF) && defined(HAVE_SETSOCKOPT)
    int size = 4194304;
    int optname = read ? SO_RCVBUF : SO_SNDBUF;

#ifdef HAVE_GETSOCKOPT
    int curr=0;
    socklen_t optlen;

    optlen = sizeof(curr);
    if(getsockopt(fd, SOL_SOCKET, optname, (void *)&curr, &optlen) == 0) {
        if(curr >= size) {
            /* Already large enough */
            return;
        }
    }
#endif /* HAVE_GETSOCKOPT */

    while(size >= 131072 &&
	  setsockopt(fd, SOL_SOCKET, optname, (void *)&size, sizeof(size)) < 0)
	size /= 2;
#endif
}


