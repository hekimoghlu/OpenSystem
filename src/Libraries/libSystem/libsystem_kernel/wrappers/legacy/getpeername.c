/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#ifndef NO_SYSCALL_LEGACY

#define _NONSTD_SOURCE
#include <sys/cdefs.h>

/*
 * We need conformance on so that EOPNOTSUPP=102.  But the routine symbol
 * will still be the legacy (undecorated) one.
 */
#undef __DARWIN_UNIX03
#define __DARWIN_UNIX03 1

#include <sys/socket.h>
#include "_errno.h"

extern int __getpeername(int, struct sockaddr * __restrict, socklen_t * __restrict);

/*
 * getpeername stub, legacy version
 */
int
getpeername(int socket, struct sockaddr * __restrict address,
    socklen_t * __restrict address_len)
{
	int ret = __getpeername(socket, address, address_len);

	/* use ENOTSUP for legacy behavior */
	if (ret < 0 && errno == EOPNOTSUPP) {
		errno = ENOTSUP;
	}
	return ret;
}

#endif /* __DARWIN_ONLY_UNIX_CONFORMANCE */
