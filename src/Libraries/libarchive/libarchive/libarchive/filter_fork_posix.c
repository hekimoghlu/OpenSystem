/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "archive_platform.h"

/* This capability is only available on POSIX systems. */
#if defined(HAVE_PIPE) && defined(HAVE_FCNTL) && \
    (defined(HAVE_FORK) || defined(HAVE_VFORK) || defined(HAVE_POSIX_SPAWNP))

#if defined(HAVE_SYS_TYPES_H)
#  include <sys/types.h>
#endif
#ifdef HAVE_ERRNO_H
#  include <errno.h>
#endif
#ifdef HAVE_STRING_H
#  include <string.h>
#endif
#if defined(HAVE_POLL) && (defined(HAVE_POLL_H) || defined(HAVE_SYS_POLL_H))
#  if defined(HAVE_POLL_H)
#    include <poll.h>
#  elif defined(HAVE_SYS_POLL_H)
#    include <sys/poll.h>
#  endif
#elif defined(HAVE_SELECT)
#  if defined(HAVE_SYS_SELECT_H)
#    include <sys/select.h>
#  elif defined(HAVE_UNISTD_H)
#    include <unistd.h>
#  endif
#endif
#ifdef HAVE_FCNTL_H
#  include <fcntl.h>
#endif
#ifdef HAVE_SPAWN_H
#  include <spawn.h>
#endif
#ifdef HAVE_STDLIB_H
#  include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif

#include "archive.h"
#include "archive_cmdline_private.h"

#include "filter_fork.h"

int
__archive_create_child(const char *cmd, int *child_stdin, int *child_stdout,
		pid_t *out_child)
{
	pid_t child = -1;
	int stdin_pipe[2], stdout_pipe[2], tmp;
#if HAVE_POSIX_SPAWNP
	posix_spawn_file_actions_t actions;
	int r;
#endif
	struct archive_cmdline *cmdline;

	cmdline = __archive_cmdline_allocate();
	if (cmdline == NULL)
		goto state_allocated;
	if (__archive_cmdline_parse(cmdline, cmd) != ARCHIVE_OK)
		goto state_allocated;

	if (pipe(stdin_pipe) == -1)
		goto state_allocated;
	if (stdin_pipe[0] == 1 /* stdout */) {
		if ((tmp = dup(stdin_pipe[0])) == -1)
			goto stdin_opened;
		close(stdin_pipe[0]);
		stdin_pipe[0] = tmp;
	}
	if (pipe(stdout_pipe) == -1)
		goto stdin_opened;
	if (stdout_pipe[1] == 0 /* stdin */) {
		if ((tmp = dup(stdout_pipe[1])) == -1)
			goto stdout_opened;
		close(stdout_pipe[1]);
		stdout_pipe[1] = tmp;
	}

#if HAVE_POSIX_SPAWNP

	r = posix_spawn_file_actions_init(&actions);
	if (r != 0) {
		errno = r;
		goto stdout_opened;
	}
	r = posix_spawn_file_actions_addclose(&actions, stdin_pipe[1]);
	if (r != 0)
		goto actions_inited;
	r = posix_spawn_file_actions_addclose(&actions, stdout_pipe[0]);
	if (r != 0)
		goto actions_inited;
	/* Setup for stdin. */
	r = posix_spawn_file_actions_adddup2(&actions, stdin_pipe[0], 0);
	if (r != 0)
		goto actions_inited;
	if (stdin_pipe[0] != 0 /* stdin */) {
		r = posix_spawn_file_actions_addclose(&actions, stdin_pipe[0]);
		if (r != 0)
			goto actions_inited;
	}
	/* Setup for stdout. */
	r = posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], 1);
	if (r != 0)
		goto actions_inited;
	if (stdout_pipe[1] != 1 /* stdout */) {
		r = posix_spawn_file_actions_addclose(&actions, stdout_pipe[1]);
		if (r != 0)
			goto actions_inited;
	}
	r = posix_spawnp(&child, cmdline->path, &actions, NULL,
		cmdline->argv, NULL);
	if (r != 0)
		goto actions_inited;
	posix_spawn_file_actions_destroy(&actions);

#else /* HAVE_POSIX_SPAWNP */

#if HAVE_VFORK
	child = vfork();
#else
	child = fork();
#endif
	if (child == -1)
		goto stdout_opened;
	if (child == 0) {
		close(stdin_pipe[1]);
		close(stdout_pipe[0]);
		if (dup2(stdin_pipe[0], 0 /* stdin */) == -1)
			_exit(254);
		if (stdin_pipe[0] != 0 /* stdin */)
			close(stdin_pipe[0]);
		if (dup2(stdout_pipe[1], 1 /* stdout */) == -1)
			_exit(254);
		if (stdout_pipe[1] != 1 /* stdout */)
			close(stdout_pipe[1]);
		execvp(cmdline->path, cmdline->argv);
		_exit(254);
	}
#endif /* HAVE_POSIX_SPAWNP */

	close(stdin_pipe[0]);
	close(stdout_pipe[1]);

	*child_stdin = stdin_pipe[1];
	fcntl(*child_stdin, F_SETFL, O_NONBLOCK);
	*child_stdout = stdout_pipe[0];
	fcntl(*child_stdout, F_SETFL, O_NONBLOCK);
	__archive_cmdline_free(cmdline);

	*out_child = child;
	return ARCHIVE_OK;

#if HAVE_POSIX_SPAWNP
actions_inited:
	errno = r;
	posix_spawn_file_actions_destroy(&actions);
#endif
stdout_opened:
	close(stdout_pipe[0]);
	close(stdout_pipe[1]);
stdin_opened:
	close(stdin_pipe[0]);
	close(stdin_pipe[1]);
state_allocated:
	__archive_cmdline_free(cmdline);
	return ARCHIVE_FAILED;
}

void
__archive_check_child(int in, int out)
{
#if defined(HAVE_POLL) && (defined(HAVE_POLL_H) || defined(HAVE_SYS_POLL_H))
	struct pollfd fds[2];
	int idx;

	idx = 0;
	if (in != -1) {
		fds[idx].fd = in;
		fds[idx].events = POLLOUT;
		++idx;
	}
	if (out != -1) {
		fds[idx].fd = out;
		fds[idx].events = POLLIN;
		++idx;
	}

	poll(fds, idx, -1); /* -1 == INFTIM, wait forever */
#elif defined(HAVE_SELECT)
	fd_set fds_in, fds_out, fds_error;

	FD_ZERO(&fds_in);
	FD_ZERO(&fds_out);
	FD_ZERO(&fds_error);
	if (out != -1) {
		FD_SET(out, &fds_in);
		FD_SET(out, &fds_error);
	}
	if (in != -1) {
		FD_SET(in, &fds_out);
		FD_SET(in, &fds_error);
	}
	select(in < out ? out + 1 : in + 1, &fds_in, &fds_out, &fds_error, NULL);
#else
	sleep(1);
#endif
}

#endif /* defined(HAVE_PIPE) && defined(HAVE_VFORK) && defined(HAVE_FCNTL) */
