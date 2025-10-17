/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <stringops.h>
#include <mymalloc.h>

/* Global library. */

#include "mail_params.h"
#include "mail_run.h"

/* mail_run_foreground - run command in foreground */

int     mail_run_foreground(const char *dir, char **argv)
{
    int     count;
    char   *path;
    WAIT_STATUS_T status;
    int     pid;
    int     wpid;

#define RETURN(x) { myfree(path); return(x); }

    path = concatenate(dir, "/", argv[0], (char *) 0);

    for (count = 0; count < var_fork_tries; count++) {
	switch (pid = fork()) {
	case -1:
	    msg_warn("fork %s: %m", path);
	    break;
	case 0:
	    /* Reset the msg_cleanup() handlers in the child process. */
	    (void) msg_cleanup((MSG_CLEANUP_FN) 0);
	    execv(path, argv);
	    msg_fatal("execv %s: %m", path);
	default:
	    do {
		wpid = waitpid(pid, &status, 0);
	    } while (wpid == -1 && errno == EINTR);
	    RETURN(wpid == -1 ? -1 :
		   WIFEXITED(status) ? WEXITSTATUS(status) : 1)
	}
	sleep(var_fork_delay);
    }
    RETURN(-1);
}

/* mail_run_background - run command in background */

int     mail_run_background(const char *dir, char **argv)
{
    int     count;
    char   *path;
    int     pid;

#define RETURN(x) { myfree(path); return(x); }

    path = concatenate(dir, "/", argv[0], (char *) 0);

    for (count = 0; count < var_fork_tries; count++) {
	switch (pid = fork()) {
	case -1:
	    msg_warn("fork %s: %m", path);
	    break;
	case 0:
	    /* Reset the msg_cleanup() handlers in the child process. */
	    (void) msg_cleanup((MSG_CLEANUP_FN) 0);
	    execv(path, argv);
	    msg_fatal("execv %s: %m", path);
	default:
	    RETURN(pid);
	}
	sleep(var_fork_delay);
    }
    RETURN(-1);
}

/* mail_run_replace - run command, replacing current process */

NORETURN mail_run_replace(const char *dir, char **argv)
{
    char   *path;

    path = concatenate(dir, "/", argv[0], (char *) 0);
    execv(path, argv);
    msg_fatal("execv %s: %m", path);
}
