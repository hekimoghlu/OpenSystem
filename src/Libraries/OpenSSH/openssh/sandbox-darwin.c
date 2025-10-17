/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#include "includes.h"

#ifdef SANDBOX_DARWIN

#include <sys/types.h>

#include <sandbox.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "log.h"
#include "ssh-sandbox.h"
#include "monitor.h"
#include "xmalloc.h"

/* Darwin/OS X sandbox */

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
	debug3("%s: preparing Darwin sandbox", __func__);
	box = xcalloc(1, sizeof(*box));
	box->child_pid = 0;

	return box;
}

void
ssh_sandbox_child(struct ssh_sandbox *box)
{
	char *errmsg;
	struct rlimit rl_zero;

	debug3("%s: starting Darwin sandbox", __func__);
	if (sandbox_init("com.openssh.sshd", SANDBOX_NAMED,
	    &errmsg) == -1)
		fatal("%s: sandbox_init: %s", __func__, errmsg);

	/*
	 * The kSBXProfilePureComputation still allows sockets, so
	 * we must disable these using rlimit.
	 */
	rl_zero.rlim_cur = rl_zero.rlim_max = 0;
	if (setrlimit(RLIMIT_FSIZE, &rl_zero) == -1)
		fatal("%s: setrlimit(RLIMIT_FSIZE, { 0, 0 }): %s",
			__func__, strerror(errno));
	if (setrlimit(RLIMIT_NOFILE, &rl_zero) == -1)
		fatal("%s: setrlimit(RLIMIT_NOFILE, { 0, 0 }): %s",
			__func__, strerror(errno));
	if (setrlimit(RLIMIT_NPROC, &rl_zero) == -1)
		fatal("%s: setrlimit(RLIMIT_NPROC, { 0, 0 }): %s",
			__func__, strerror(errno));
}

void
ssh_sandbox_parent_finish(struct ssh_sandbox *box)
{
	free(box);
	debug3("%s: finished", __func__);
}

void
ssh_sandbox_parent_preauth(struct ssh_sandbox *box, pid_t child_pid)
{
	box->child_pid = child_pid;
}

#endif /* SANDBOX_DARWIN */
