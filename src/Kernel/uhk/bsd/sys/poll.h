/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
/*-
 * Copyright (c) 1997 Peter Wemm <peter@freebsd.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#ifndef _SYS_POLL_H_
#define _SYS_POLL_H_

/*
 * This file is intended to be compatible with the traditional poll.h.
 */

/*
 * Requestable events.  If poll(2) finds any of these set, they are
 * copied to revents on return.
 */
#define POLLIN          0x0001          /* any readable data available */
#define POLLPRI         0x0002          /* OOB/Urgent readable data */
#define POLLOUT         0x0004          /* file descriptor is writeable */
#define POLLRDNORM      0x0040          /* non-OOB/URG data available */
#define POLLWRNORM      POLLOUT         /* no write type differentiation */
#define POLLRDBAND      0x0080          /* OOB/Urgent readable data */
#define POLLWRBAND      0x0100          /* OOB/Urgent data can be written */

/*
 * FreeBSD extensions: polling on a regular file might return one
 * of these events (currently only supported on local filesystems).
 */
#define POLLEXTEND      0x0200          /* file may have been extended */
#define POLLATTRIB      0x0400          /* file attributes may have changed */
#define POLLNLINK       0x0800          /* (un)link/rename may have happened */
#define POLLWRITE       0x1000          /* file's contents may have changed */

/*
 * These events are set if they occur regardless of whether they were
 * requested.
 */
#define POLLERR         0x0008          /* some poll error occurred */
#define POLLHUP         0x0010          /* file descriptor was "hung up" */
#define POLLNVAL        0x0020          /* requested events "invalid" */

#define POLLSTANDARD    (POLLIN|POLLPRI|POLLOUT|POLLRDNORM|POLLRDBAND|\
	                 POLLWRBAND|POLLERR|POLLHUP|POLLNVAL)

struct pollfd {
	int     fd;
	short   events;
	short   revents;
};

typedef unsigned int nfds_t;

#if !defined(KERNEL)

#include <sys/cdefs.h>

__BEGIN_DECLS

/*
 * This is defined here (instead of <poll.h>) because this is where
 * traditional SVR4 code will look to find it.
 */
extern int poll(struct pollfd *, nfds_t, int) __DARWIN_ALIAS_C(poll);

__END_DECLS

#endif /* !KERNEL */

#endif /* !_SYS_POLL_H_ */
