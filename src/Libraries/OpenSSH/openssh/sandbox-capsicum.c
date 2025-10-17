/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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

#ifdef SANDBOX_CAPSICUM

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/capsicum.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef HAVE_CAPSICUM_HELPERS_H
#include <capsicum_helpers.h>
#endif

#include "log.h"
#include "monitor.h"
#include "ssh-sandbox.h"
#include "xmalloc.h"

/*
 * Capsicum sandbox that sets zero nfiles, nprocs and filesize rlimits,
 * limits rights on stdout, stdin, stderr, monitor and switches to
 * capability mode.
 */

struct ssh_sandbox {
	struct monitor *monitor;
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
	debug3("%s: preparing capsicum sandbox", __func__);
	box = xcalloc(1, sizeof(*box));
	box->monitor = monitor;
	box->child_pid = 0;

	return box;
}

void
ssh_sandbox_child(struct ssh_sandbox *box)
{
	struct rlimit rl_zero;
	cap_rights_t rights;

#ifdef HAVE_CAPH_CACHE_TZDATA
	caph_cache_tzdata();
#endif

	rl_zero.rlim_cur = rl_zero.rlim_max = 0;

	if (setrlimit(RLIMIT_FSIZE, &rl_zero) == -1)
		fatal("%s: setrlimit(RLIMIT_FSIZE, { 0, 0 }): %s",
			__func__, strerror(errno));
#ifndef SANDBOX_SKIP_RLIMIT_NOFILE
	if (setrlimit(RLIMIT_NOFILE, &rl_zero) == -1)
		fatal("%s: setrlimit(RLIMIT_NOFILE, { 0, 0 }): %s",
			__func__, strerror(errno));
#endif
	if (setrlimit(RLIMIT_NPROC, &rl_zero) == -1)
		fatal("%s: setrlimit(RLIMIT_NPROC, { 0, 0 }): %s",
			__func__, strerror(errno));

	cap_rights_init(&rights);

	if (cap_rights_limit(STDIN_FILENO, &rights) < 0 && errno != ENOSYS)
		fatal("can't limit stdin: %m");
	if (cap_rights_limit(STDOUT_FILENO, &rights) < 0 && errno != ENOSYS)
		fatal("can't limit stdout: %m");
	if (cap_rights_limit(STDERR_FILENO, &rights) < 0 && errno != ENOSYS)
		fatal("can't limit stderr: %m");

	cap_rights_init(&rights, CAP_READ, CAP_WRITE);
	if (cap_rights_limit(box->monitor->m_recvfd, &rights) < 0 &&
	    errno != ENOSYS)
		fatal("%s: failed to limit the network socket", __func__);
	cap_rights_init(&rights, CAP_WRITE);
	if (cap_rights_limit(box->monitor->m_log_sendfd, &rights) < 0 &&
	    errno != ENOSYS)
		fatal("%s: failed to limit the logging socket", __func__);
	if (cap_enter() < 0 && errno != ENOSYS)
		fatal("%s: failed to enter capability mode", __func__);

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

#endif /* SANDBOX_CAPSICUM */
