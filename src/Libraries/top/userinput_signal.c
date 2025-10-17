/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#include "preferences.h"
#include "userinput.h"
#include <ctype.h>
#include <curses.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static bool
is_empty(const char *s)
{

	while (*s) {
		if (!isspace(*s)) {
			return false;
		}

		++s;
	}

	return true;
}

static bool
is_pid(const char *s)
{
	const char *sp;

	for (sp = s; *sp; ++sp) {
		if (!isdigit(*sp)) {
			return false;
		}
	}

	/* If the string was not an empty string, then it contains digits. */
	if (sp != s) {
		return true;
	}

	return false;
}

static void
reset_pid(struct user_input_state *s)
{
	s->buf[0] = '\0';
	s->offset = 0;
}

static void
signal_pid_completion(void *tinst, struct user_input_state *s)
{
	const char *signame;
	int sig;
	int err;
	uid_t euid;
	gid_t egid;
	int saved_errno = 0;
	pid_t pid;

	if (is_empty(s->buf)) {
		/*
		 * Any empty buffer indicates that the user didn't want
		 * to signal the process after all.
		 */

		reset_pid(s);
		user_input_set_state(NULL);
		return;
	}

	if (!is_pid(s->buf)) {
		reset_pid(s);
		user_input_set_error_state("invalid pid");
		return;
	}

	pid = atoi(s->buf);

	reset_pid(s);

	sig = top_prefs_get_signal(&signame);

	/* Temporarily drop permissions. */
	euid = geteuid();
	egid = getegid();

	if (-1 == seteuid(getuid()) || -1 == setegid(getgid())) {
		user_input_set_error_state("missing setuid bit");
		return;
	}

	err = kill(pid, sig);

	if (-1 == err)
		saved_errno = errno;

	if (-1 == seteuid(euid) || -1 == setegid(egid)) {
		user_input_set_error_state("restoring setuid bit");
		return;
	}

	switch (saved_errno) {
	case EINVAL:
		user_input_set_error_state("invalid signal");
		return;

	case ESRCH:
		user_input_set_error_state("invalid pid");
		return;

	case EPERM:
		user_input_set_error_state("permission error signaling");
		return;
	}

	user_input_set_state(NULL);
}

static void
signal_pid_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	char display[60];

	if (-1 == snprintf(display, sizeof(display), "pid: %s", s->buf)) {
		user_input_set_error_state("string input too long!");
		return;
	}

	mvwaddstr(win, row, column, display);
}

struct user_input_state top_user_input_signal_pid_state
		= { .offset = 0, .completion = signal_pid_completion, .draw = signal_pid_draw };

static void
signal_completion(void *tinst, struct user_input_state *s)
{

	if (!strlen(s->buf)) {
		/* Use the current default. */
		user_input_set_state(&top_user_input_signal_pid_state);
		return;
	}

	if (top_prefs_set_signal_string(s->buf)) {
		user_input_set_error_state("invalid signal name");
		return;
	}

	user_input_set_state(&top_user_input_signal_pid_state);
}

static void
signal_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	char display[60];
	const char *signame;

	(void)top_prefs_get_signal(&signame);

	if (-1 == snprintf(display, sizeof(display), "signal [%s]: %s", signame, s->buf)) {
		user_input_set_error_state("string input too long!");
		return;
	}

	mvwaddstr(win, row, column, display);
}

struct user_input_state top_user_input_signal_state
		= { .offset = 0, .completion = signal_completion, .draw = signal_draw };
