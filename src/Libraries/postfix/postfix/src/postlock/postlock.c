/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include <sys/stat.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <msg_vstream.h>
#include <iostuff.h>
#include <warn_stat.h>
#include <clean_env.h>

/* Global library. */

#include <mail_params.h>
#include <mail_version.h>
#include <dot_lockfile.h>
#include <deliver_flock.h>
#include <mail_conf.h>
#include <sys_exits.h>
#include <mbox_conf.h>
#include <mbox_open.h>
#include <dsn_util.h>
#include <mail_parm_split.h>

/* Application-specific. */

/* usage - explain */

static NORETURN usage(char *myname)
{
    msg_fatal("usage: %s [-c config_dir] [-l lock_style] [-v] folder command...", myname);
}

/* fatal_exit - all failures are deemed recoverable */

static void fatal_exit(void)
{
    exit(EX_TEMPFAIL);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - go for it */

int     main(int argc, char **argv)
{
    DSN_BUF *why;
    char   *folder;
    char  **command;
    int     ch;
    int     fd;
    struct stat st;
    int     count;
    WAIT_STATUS_T status;
    pid_t   pid;
    int     lock_mask;
    char   *lock_style = 0;
    MBOX   *mp;
    ARGV   *import_env;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Be consistent with file permissions.
     */
    umask(022);

    /*
     * To minimize confusion, make sure that the standard file descriptors
     * are open before opening anything else. XXX Work around for 44BSD where
     * fstat can return EBADF on an open file descriptor.
     */
    for (fd = 0; fd < 3; fd++)
	if (fstat(fd, &st) == -1
	    && (close(fd), open("/dev/null", O_RDWR, 0)) != fd)
	    msg_fatal("open /dev/null: %m");

    /*
     * Process environment options as early as we can. We are not set-uid,
     * and we are supposed to be running in a controlled environment.
     */
    if (getenv(CONF_ENV_VERB))
	msg_verbose = 1;

    /*
     * Set up logging and error handling. Intercept fatal exits so we can
     * return a distinguished exit status.
     */
    msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_cleanup(fatal_exit);

    /*
     * Parse JCL.
     */
    while ((ch = GETOPT(argc, argv, "c:l:v")) > 0) {
	switch (ch) {
	default:
	    usage(argv[0]);
	    break;
	case 'c':
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal("out of memory");
	    break;
	case 'l':
	    lock_style = optarg;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	}
    }
    if (optind + 2 > argc)
	usage(argv[0]);
    folder = argv[optind];
    command = argv + optind + 1;

    /*
     * Read the config file. The command line lock style can override the
     * configured lock style.
     */
    mail_conf_read();
    /* Enforce consistent operation of different Postfix parts. */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    update_env(import_env->argv);
    argv_free(import_env);
    lock_mask = mbox_lock_mask(lock_style ? lock_style :
	       get_mail_conf_str(VAR_MAILBOX_LOCK, DEF_MAILBOX_LOCK, 1, 0));

    /*
     * Lock the folder for exclusive access. Lose the lock upon exit. The
     * command is not supposed to disappear into the background.
     */
    why = dsb_create();
    if ((mp = mbox_open(folder, O_APPEND | O_WRONLY | O_CREAT,
			S_IRUSR | S_IWUSR, (struct stat *) 0,
			-1, -1, lock_mask, "5.2.0", why)) == 0)
	msg_fatal("open file %s: %s", folder, vstring_str(why->reason));
    dsb_free(why);

    /*
     * Run the command. Remove the lock after completion.
     */
    for (count = 1; (pid = fork()) == -1; count++) {
	msg_warn("fork %s: %m", command[0]);
	if (count >= var_fork_tries) {
	    mbox_release(mp);
	    exit(EX_TEMPFAIL);
	}
	sleep(var_fork_delay);
    }
    switch (pid) {
    case 0:
	(void) msg_cleanup((MSG_CLEANUP_FN) 0);
	execvp(command[0], command);
	msg_fatal("execvp %s: %m", command[0]);
    default:
	if (waitpid(pid, &status, 0) < 0)
	    msg_fatal("waitpid: %m");
	vstream_fclose(mp->fp);
	mbox_release(mp);
	exit(WIFEXITED(status) ? WEXITSTATUS(status) : 1);
    }
}
