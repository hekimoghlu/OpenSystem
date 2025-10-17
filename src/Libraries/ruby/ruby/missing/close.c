/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include "ruby/missing.h"
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>

#undef getpeername
int
ruby_getpeername(int s, struct sockaddr * name,
         socklen_t * namelen)
{
    int err = errno;
    errno = 0;
    s = getpeername(s, name, namelen);
    if (errno == ECONNRESET) {
	errno = 0;
	s = 0;
    }
    else if (errno == 0)
	errno = err;
    return s;
}

#undef getsockname
int
ruby_getsockname(int s, struct sockaddr * name,
         socklen_t * namelen)
{
    int err = errno;
    errno = 0;
    s = getsockname(s, name, namelen);
    if (errno == ECONNRESET) {
	errno = 0;
	s = 0;
    }
    else if (errno == 0)
	errno = err;
    return s;
}

#undef shutdown
int
ruby_shutdown(int s, int how)
{
    int err = errno;
    errno = 0;
    s = shutdown(s, how);
    if (errno == ECONNRESET) {
	errno = 0;
	s = 0;
    }
    else if (errno == 0)
	errno = err;
    return s;
}

#undef close
int
ruby_close(int s)
{
    int err = errno;
    errno = 0;
    s = close(s);
    if (errno == ECONNRESET) {
	errno = 0;
	s = 0;
    }
    else if (errno == 0)
	errno = err;
    return s;
}
