/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
/*
 * Copyright (c) 2011 Damien Miller <djm@mindrot.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "includes.h"

#ifdef SANDBOX_RLIMIT

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "log.h"
#include "ssh-sandbox.h"
#include "xmalloc.h"

/* Minimal sandbox that sets zero nfiles, nprocs and filesize rlimits */

struct ssh_sandbox {
	pid_t child_pid;
};

struct ssh_sandbox *
ssh_sandbox_init(struct monitor *monitor)
{
	struct ssh_sandbox *box;

	/*
	 * Strictly, we don't need to maintain any state here but we need
	 * to return non-NULL to satisfy the API.
	 */
	debug3_f("preparing rlimit sandbox");
	box = xcalloc(1, sizeof(*box));
	box->child_pid = 0;

	return box;
}

void
ssh_sandbox_child(struct ssh_sandbox *box)
{
	struct rlimit rl_zero;

	rl_zero.rlim_cur = rl_zero.rlim_max = 0;

#ifndef SANDBOX_SKIP_RLIMIT_FSIZE
	if (setrlimit(RLIMIT_FSIZE, &rl_zero) == -1)
		fatal_f("setrlimit(RLIMIT_FSIZE, { 0, 0 }): %s",
			strerror(errno));
#endif
#ifndef SANDBOX_SKIP_RLIMIT_NOFILE
	if (setrlimit(RLIMIT_NOFILE, &rl_zero) == -1)
		fatal_f("setrlimit(RLIMIT_NOFILE, { 0, 0 }): %s",
			strerror(errno));
#endif
#ifdef HAVE_RLIMIT_NPROC
	if (setrlimit(RLIMIT_NPROC, &rl_zero) == -1)
		fatal_f("setrlimit(RLIMIT_NPROC, { 0, 0 }): %s",
			strerror(errno));
#endif
}

void
ssh_sandbox_parent_finish(struct ssh_sandbox *box)
{
	free(box);
	debug3_f("finished");
}

void
ssh_sandbox_parent_preauth(struct ssh_sandbox *box, pid_t child_pid)
{
	box->child_pid = child_pid;
}

#endif /* SANDBOX_RLIMIT */
