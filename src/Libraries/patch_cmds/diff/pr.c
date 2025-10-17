/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#ifndef __APPLE__
#include <sys/procdesc.h>
#endif
#include <sys/wait.h>

#include <err.h>
#ifdef __APPLE__
#include <errno.h>
#endif
#include <paths.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "pr.h"
#include "diff.h"
#include "xmalloc.h"

#define _PATH_PR "/usr/bin/pr"

struct pr *
start_pr(char *file1, char *file2)
{
	int pfd[2];
#ifndef __APPLE__
	int pr_pd;
#endif
	pid_t pid;
	char *header;
	struct pr *pr;

	pr = xcalloc(1, sizeof(*pr));

	xasprintf(&header, "%s %s %s", diffargs, file1, file2);
	signal(SIGPIPE, SIG_IGN);
	fflush(stdout);
	rewind(stdout);
	if (pipe(pfd) == -1)
		err(2, "pipe");
#ifdef __APPLE__
	switch ((pid = fork())) {
#else
	switch ((pid = pdfork(&pr_pd, PD_CLOEXEC))) {
#endif
	case -1:
		status |= 2;
		free(header);
		err(2, "No more processes");
	case 0:
		/* child */
		if (pfd[0] != STDIN_FILENO) {
			dup2(pfd[0], STDIN_FILENO);
			close(pfd[0]);
		}
		close(pfd[1]);
		execl(_PATH_PR, _PATH_PR, "-h", header, (char *)0);
		_exit(127);
	default:

		/* parent */
		if (pfd[1] != STDOUT_FILENO) {
			pr->ostdout = dup(STDOUT_FILENO);
			dup2(pfd[1], STDOUT_FILENO);
			close(pfd[1]);
		}
		close(pfd[0]);
		rewind(stdout);
		free(header);
		pr->kq = kqueue();
		if (pr->kq == -1)
			err(2, "kqueue");
		pr->e = xmalloc(sizeof(struct kevent));
#ifdef __APPLE__
		EV_SET(pr->e, pid, EVFILT_PROC, EV_ADD,
		    NOTE_EXIT | NOTE_EXITSTATUS, 0, NULL);
#else
		EV_SET(pr->e, pr_pd, EVFILT_PROCDESC, EV_ADD, NOTE_EXIT, 0,
		    NULL);
#endif
		if (kevent(pr->kq, pr->e, 1, NULL, 0, NULL) == -1)
			err(2, "kevent");
	}
	return (pr);
}

/* close the pipe to pr and restore stdout */
void
stop_pr(struct pr *pr)
{
	int wstatus;

	if (pr == NULL)
		return;

	fflush(stdout);
	if (pr->ostdout != STDOUT_FILENO) {
		close(STDOUT_FILENO);
		dup2(pr->ostdout, STDOUT_FILENO);
		close(pr->ostdout);
	}
	if (kevent(pr->kq, NULL, 0, pr->e, 1, NULL) == -1)
		err(2, "kevent");
	wstatus = pr->e[0].data;
#ifdef __APPLE__
	/* Reap it. */
	(void)waitpid((pid_t)pr->e[0].ident, NULL, WNOHANG);
#endif
	close(pr->kq);
	free(pr);
	if (WIFEXITED(wstatus) && WEXITSTATUS(wstatus) != 0)
		errx(2, "pr exited abnormally");
	else if (WIFSIGNALED(wstatus))
		errx(2, "pr killed by signal %d",
		    WTERMSIG(wstatus));
}
