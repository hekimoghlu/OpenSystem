/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#include <sys/types.h>

#include <ctype.h>
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

int openpam_ttyconv_timeout = 0;

static void
timeout(int sig)
{

	(void)sig;
}

static char *
prompt(const char *msg)
{
	char buf[PAM_MAX_RESP_SIZE];
	struct sigaction action, saved_action;
	sigset_t saved_sigset, sigset;
	unsigned int saved_alarm;
	int eof, error, fd;
	size_t len;
	char *retval;
	char ch;

	sigemptyset(&sigset);
	sigaddset(&sigset, SIGINT);
	sigaddset(&sigset, SIGTSTP);
	sigprocmask(SIG_SETMASK, &sigset, &saved_sigset);
	action.sa_handler = &timeout;
	action.sa_flags = 0;
	sigemptyset(&action.sa_mask);
	sigaction(SIGALRM, &action, &saved_action);
	fputs(msg, stdout);
	fflush(stdout);
#ifdef HAVE_FPURGE
	fpurge(stdin);
#endif
	fd = fileno(stdin);
	buf[0] = '\0';
	eof = error = 0;
	saved_alarm = 0;
	if (openpam_ttyconv_timeout >= 0)
		saved_alarm = alarm(openpam_ttyconv_timeout);
	ch = '\0';
	for (len = 0; ch != '\n' && !eof && !error; ++len) {
		switch (read(fd, &ch, 1)) {
		case 1:
			if (len < PAM_MAX_RESP_SIZE - 1) {
				buf[len + 1] = '\0';
				buf[len] = ch;
			}
			break;
		case 0:
			eof = 1;
			break;
		default:
			error = errno;
			break;
		}
	}
	if (openpam_ttyconv_timeout >= 0)
		alarm(0);
	sigaction(SIGALRM, &saved_action, NULL);
	sigprocmask(SIG_SETMASK, &saved_sigset, NULL);
	if (saved_alarm > 0)
		alarm(saved_alarm);
	if (error == EINTR)
		fputs(" timeout!", stderr);
	if (error || eof) {
		fputs("\n", stderr);
		memset(buf, 0, sizeof(buf));
		return (NULL);
	}
	/* trim trailing whitespace */
	for (len = strlen(buf); len > 0; --len)
		if (buf[len - 1] != '\r' && buf[len - 1] != '\n')
			break;
	buf[len] = '\0';
	retval = strdup(buf);
	memset(buf, 0, sizeof(buf));
	return (retval);
}

static char *
prompt_echo_off(const char *msg)
{
	struct termios tattr;
	tcflag_t lflag;
	char *ret;
	int fd;

	fd = fileno(stdin);
	if (tcgetattr(fd, &tattr) != 0) {
		openpam_log(PAM_LOG_ERROR, "tcgetattr(): %m");
		return (NULL);
	}
	lflag = tattr.c_lflag;
	tattr.c_lflag &= ~ECHO;
	if (tcsetattr(fd, TCSAFLUSH, &tattr) != 0) {
		openpam_log(PAM_LOG_ERROR, "tcsetattr(): %m");
		return (NULL);
	}
	ret = prompt(msg);
	tattr.c_lflag = lflag;
	(void)tcsetattr(fd, TCSANOW, &tattr);
	if (ret != NULL)
		fputs("\n", stdout);
	return (ret);
}

/*
 * OpenPAM extension
 *
 * Simple tty-based conversation function
 */

int
openpam_ttyconv(int n,
	 const struct pam_message **msg,
	 struct pam_response **resp,
	 void *data)
{
	struct pam_response *aresp;
	int i;

	ENTER();
	(void)data;
	if (n <= 0 || n > PAM_MAX_NUM_MSG)
		RETURNC(PAM_CONV_ERR);
	if ((aresp = calloc(n, sizeof *aresp)) == NULL)
		RETURNC(PAM_BUF_ERR);
	for (i = 0; i < n; ++i) {
		aresp[i].resp_retcode = 0;
		aresp[i].resp = NULL;
		switch (msg[i]->msg_style) {
		case PAM_PROMPT_ECHO_OFF:
			aresp[i].resp = prompt_echo_off(msg[i]->msg);
			if (aresp[i].resp == NULL)
				goto fail;
			break;
		case PAM_PROMPT_ECHO_ON:
			aresp[i].resp = prompt(msg[i]->msg);
			if (aresp[i].resp == NULL)
				goto fail;
			break;
		case PAM_ERROR_MSG:
			fputs(msg[i]->msg, stderr);
			if (strlen(msg[i]->msg) > 0 &&
			    msg[i]->msg[strlen(msg[i]->msg) - 1] != '\n')
				fputc('\n', stderr);
			break;
		case PAM_TEXT_INFO:
			fputs(msg[i]->msg, stdout);
			if (strlen(msg[i]->msg) > 0 &&
			    msg[i]->msg[strlen(msg[i]->msg) - 1] != '\n')
				fputc('\n', stdout);
			break;
		default:
			goto fail;
		}
	}
	*resp = aresp;
	RETURNC(PAM_SUCCESS);
 fail:
	for (i = 0; i < n; ++i) {
		if (aresp[i].resp != NULL) {
			memset(aresp[i].resp, 0, strlen(aresp[i].resp));
			FREE(aresp[i].resp);
		}
	}
	memset(aresp, 0, n * sizeof *aresp);
	FREE(aresp);
	*resp = NULL;
	RETURNC(PAM_CONV_ERR);
}

/*
 * XXX OpenPAM migration hack
 *
 * misc_conv
 */
 
int misc_conv(int num_msg,
	const struct pam_message **msgm,
	struct pam_response **response,
	void *appdata_ptr)
{
	return openpam_ttyconv(num_msg, msgm, response, appdata_ptr);
}


/*
 * Error codes:
 *
 *	PAM_SYSTEM_ERR
 *	PAM_BUF_ERR
 *	PAM_CONV_ERR
 */

/**
 * The =openpam_ttyconv function is a standard conversation function
 * suitable for use on TTY devices.
 * It should be adequate for the needs of most text-based interactive
 * programs.
 *
 * The =openpam_ttyconv function allows the application to specify a
 * timeout for user input by setting the global integer variable
 * :openpam_ttyconv_timeout to the length of the timeout in seconds.
 *
 * >openpam_nullconv
 * >pam_prompt
 * >pam_vprompt
 */
